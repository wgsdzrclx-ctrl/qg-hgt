"""Quality--Consistency collaborative heterogeneous graph transformer (QC-HGT).

The implementation follows paper-framework Sections 4.5--4.8. Artificial
corruption parameters are never model inputs: learned reliability uses only
raw observable windows, masks, and optional timestamp-delay observations.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax

from data.tep_hetero_graph import EDGE_TYPES, PROCESS_UNITS


NODE_TYPES = ("sensor", "actuator", "process_unit")


def _relation_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


def _activation(name: str) -> nn.Module:
    choices: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
    }
    if name not in choices:
        raise ValueError(f"Unsupported activation: {name}")
    return choices[name]


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, int(hidden)), _activation(activation), nn.Dropout(dropout)))
        previous = int(hidden)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class ObservableQualityFeatures(nn.Module):
    """Compute the six observable descriptors a_j from sensor windows."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.high_frequency_start = float(config["fft_high_frequency_start"])
        self.history_length = int(config["history_length"])
        self.normalize_history = bool(config["normalize_history_deviation"])
        self.repeat_tolerance = float(config["repeat_tolerance"])
        self.timestamp_delay_scale = float(config["timestamp_delay_scale"])
        self.epsilon = float(config["epsilon"])
        if not 0.0 <= self.high_frequency_start <= 1.0:
            raise ValueError("fft_high_frequency_start must be in [0, 1].")
        if self.history_length <= 0 or self.timestamp_delay_scale <= 0.0:
            raise ValueError("history_length and timestamp_delay_scale must be positive.")

    def forward(self, raw_window: Tensor, observed_mask: Tensor, timestamp_delay: Tensor) -> Tensor:
        if raw_window.ndim != 2 or observed_mask.shape != raw_window.shape:
            raise ValueError("Quality descriptors require [sensor_nodes, time] windows and masks.")
        mask = observed_mask.bool() & torch.isfinite(raw_window)
        count = mask.sum(dim=1).to(raw_window.dtype)
        safe_count = count.clamp_min(1.0)
        mean = torch.where(mask, raw_window, torch.zeros_like(raw_window)).sum(dim=1) / safe_count
        filled = torch.where(mask, raw_window, mean[:, None])

        missing_rate = 1.0 - mask.to(raw_window.dtype).mean(dim=1)
        difference = filled[:, 1:] - filled[:, :-1]
        valid_difference = mask[:, 1:] & mask[:, :-1]
        difference_count = valid_difference.sum(dim=1).to(raw_window.dtype)
        safe_difference_count = difference_count.clamp_min(1.0)
        difference_mean = (
            torch.where(valid_difference, difference, torch.zeros_like(difference)).sum(dim=1)
            / safe_difference_count
        )
        difference_centered = torch.where(
            valid_difference,
            difference - difference_mean[:, None],
            torch.zeros_like(difference),
        )
        difference_variance = difference_centered.square().sum(dim=1) / safe_difference_count
        difference_variance = torch.where(
            difference_count > 1.0, difference_variance, torch.zeros_like(difference_variance)
        )

        centered = filled - filled.mean(dim=1, keepdim=True)
        spectrum_energy = torch.fft.rfft(centered, dim=1).abs().square()
        non_dc_energy = spectrum_energy[:, 1:]
        high_start = max(0, math.floor(non_dc_energy.shape[1] * self.high_frequency_start))
        high_energy = non_dc_energy[:, high_start:].sum(dim=1)
        total_energy = non_dc_energy.sum(dim=1)
        high_frequency_ratio = high_energy / total_energy.clamp_min(self.epsilon)
        high_frequency_ratio = torch.where(
            total_energy > self.epsilon,
            high_frequency_ratio,
            torch.zeros_like(high_frequency_ratio),
        )

        history_start = max(0, raw_window.shape[1] - 1 - self.history_length)
        history = raw_window[:, history_start:-1]
        history_mask = mask[:, history_start:-1]
        history_count = history_mask.sum(dim=1).to(raw_window.dtype)
        history_mean = (
            torch.where(history_mask, history, torch.zeros_like(history)).sum(dim=1)
            / history_count.clamp_min(1.0)
        )
        last = torch.where(mask[:, -1], raw_window[:, -1], mean)
        history_deviation = (last - history_mean).abs()
        if self.normalize_history:
            history_variance = (
                torch.where(history_mask, history - history_mean[:, None], torch.zeros_like(history))
                .square()
                .sum(dim=1)
                / history_count.clamp_min(1.0)
            )
            history_deviation = history_deviation / history_variance.sqrt().clamp_min(self.epsilon)
        history_deviation = torch.where(
            history_count > 0.0, history_deviation, torch.zeros_like(history_deviation)
        )

        comparable = mask[:, 1:] & mask[:, :-1]
        repeated = comparable & ((raw_window[:, 1:] - raw_window[:, :-1]).abs() <= self.repeat_tolerance)
        repeated_ratio = repeated.sum(dim=1).to(raw_window.dtype) / comparable.sum(dim=1).clamp_min(1)
        delay = timestamp_delay.reshape(-1).to(raw_window.dtype) / self.timestamp_delay_scale
        return torch.stack(
            (
                missing_rate,
                difference_variance,
                high_frequency_ratio,
                history_deviation,
                repeated_ratio,
                delay,
            ),
            dim=1,
        )


class QualityEstimator(nn.Module):
    """Section 4.5: r_j = sigmoid(MLP_q(a_j))."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.features = ObservableQualityFeatures(config)
        normalization = str(config["descriptor_normalization"])
        if normalization == "layer_norm":
            self.descriptor_normalization: nn.Module = nn.LayerNorm(6)
        elif normalization == "none":
            self.descriptor_normalization = nn.Identity()
        else:
            raise ValueError("quality.descriptor_normalization must be layer_norm or none.")
        self.network = _mlp(
            input_dim=6,
            hidden_dims=config["hidden_dims"],
            output_dim=1,
            activation=str(config["activation"]),
            dropout=float(config["dropout"]),
        )

    def forward(self, data: HeteroData) -> tuple[Tensor, Tensor]:
        storage = data["sensor"]
        descriptors = self.features(
            storage.raw_window,
            storage.observed_mask,
            storage.timestamp_delay,
        )
        reliability = torch.sigmoid(
            self.network(self.descriptor_normalization(descriptors))
        ).squeeze(-1)
        return reliability, descriptors


class QCHGTLayer(nn.Module):
    """One relation-aware collaborative quality--consistency propagation layer."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.hidden_dim = int(config["hidden_dim"])
        self.relation_dim = int(config["relation_dim"])
        self.epsilon = float(config["attention"]["epsilon"])
        self.negative_slope = float(config["attention"]["negative_slope"])
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.activation = _activation(str(config["activation"]))
        self.switches = config["switches"]
        self.negative_config = config["consistency"]["negative_sampling"]
        memory_config = config["consistency"]["memory"]
        self.consistency_chunk_size = int(memory_config["logit_chunk_size"])
        self.consistency_gradient_checkpointing = bool(memory_config["gradient_checkpointing"])
        if self.consistency_chunk_size <= 0:
            raise ValueError("consistency.memory.logit_chunk_size must be positive.")
        self.use_relation_types = bool(self.switches.get("relation_types", True))

        self.self_projection = nn.ModuleDict(
            {node_type: nn.Linear(self.hidden_dim, self.hidden_dim) for node_type in NODE_TYPES}
        )
        self.source_projection = nn.ModuleDict()
        self.target_projection = nn.ModuleDict()
        self.relation_projection = nn.ModuleDict()
        self.attention_vector = nn.ParameterDict()
        parameter_keys = [_relation_key(edge_type) for edge_type in EDGE_TYPES] if self.use_relation_types else ["shared"]
        for key in parameter_keys:
            self.source_projection[key] = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.target_projection[key] = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.relation_projection[key] = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.attention_vector[key] = nn.Parameter(
                torch.empty(2 * self.hidden_dim + self.relation_dim)
            )
        self.relation_embedding = nn.Parameter(torch.empty(len(EDGE_TYPES), self.relation_dim))
        consistency_input = 3 * self.hidden_dim + self.relation_dim
        self.consistency_network = _mlp(
            consistency_input,
            config["consistency"]["hidden_dims"],
            1,
            str(config["consistency"]["activation"]),
            float(config["consistency"]["dropout"]),
        )

        quality_initial = torch.tensor(config["attention"]["quality_lambda_initial"], dtype=torch.float32)
        consistency_initial = torch.tensor(config["attention"]["consistency_gamma_initial"], dtype=torch.float32)
        if quality_initial.numel() != len(EDGE_TYPES) or consistency_initial.numel() != len(EDGE_TYPES):
            raise ValueError("Attention lambda/gamma initializers must have one value per edge type.")
        if bool(config["attention"]["learnable_fusion_coefficients"]):
            self.quality_lambda = nn.Parameter(quality_initial)
            self.consistency_gamma = nn.Parameter(consistency_initial)
        else:
            self.register_buffer("quality_lambda", quality_initial)
            self.register_buffer("consistency_gamma", consistency_initial)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.relation_embedding)
        for parameter in self.attention_vector.values():
            nn.init.xavier_uniform_(parameter.unsqueeze(0))

    def _consistency_logits(
        self,
        source: Tensor,
        target: Tensor,
        relation: Tensor,
    ) -> Tensor:
        features = torch.cat((source, target, relation, (source - target).abs()), dim=-1)
        return self.consistency_network(features).squeeze(-1)

    def _indexed_consistency_logits(
        self,
        source: Tensor,
        target_all: Tensor,
        target_index: Tensor,
        relation: Tensor,
    ) -> Tensor:
        """Score indexed negative targets in memory-bounded chunks."""
        chunks: list[Tensor] = []
        for start in range(0, source.shape[0], self.consistency_chunk_size):
            stop = min(start + self.consistency_chunk_size, source.shape[0])
            source_chunk = source[start:stop]
            index_chunk = target_index[start:stop]
            relation_chunk = relation[start:stop]

            def score(
                chunk_source: Tensor,
                all_targets: Tensor,
                chunk_index: Tensor,
                chunk_relation: Tensor,
            ) -> Tensor:
                return self._consistency_logits(
                    chunk_source, all_targets[chunk_index], chunk_relation
                )

            if self.consistency_gradient_checkpointing and self.training and torch.is_grad_enabled():
                logits = checkpoint(
                    score,
                    source_chunk,
                    target_all,
                    index_chunk,
                    relation_chunk,
                    use_reentrant=False,
                )
            else:
                logits = score(source_chunk, target_all, index_chunk, relation_chunk)
            chunks.append(logits)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _cross_graph_target_indices(storage: Any, target_index: Tensor) -> Tensor | None:
        if not hasattr(storage, "batch") or not hasattr(storage, "ptr") or storage.ptr.numel() <= 2:
            return None
        graph_index = storage.batch[target_index]
        next_graph = (graph_index + 1) % (storage.ptr.numel() - 1)
        local_index = target_index - storage.ptr[graph_index]
        next_size = storage.ptr[next_graph + 1] - storage.ptr[next_graph]
        return storage.ptr[next_graph] + torch.remainder(local_index, next_size)

    @staticmethod
    def _wrong_target_indices(
        source_index: Tensor,
        target_index: Tensor,
        source_count: int,
        target_count: int,
    ) -> Tensor:
        """Sample targets that are not true neighbors of each edge source."""
        adjacency = torch.zeros(
            (source_count, target_count), dtype=torch.bool, device=source_index.device
        )
        adjacency[source_index, target_index] = True
        sampled = []
        for source in source_index.tolist():
            candidates = (~adjacency[source]).nonzero(as_tuple=False).reshape(-1)
            if candidates.numel() == 0:
                raise ValueError("Cannot construct a wrong-connection negative for a complete relation.")
            choice = torch.randint(candidates.numel(), (), device=source_index.device)
            sampled.append(candidates[choice])
        return torch.stack(sampled)

    def _consistency_loss(
        self,
        positive_logits: Tensor,
        source_projected: Tensor,
        source_index: Tensor,
        source_count: int,
        target_projected_all: Tensor,
        target_index: Tensor,
        relation: Tensor,
        target_storage: Any,
        temporal_target_all: Tensor | None,
    ) -> Tensor:
        negative_logits: list[Tensor] = []
        target_count = target_projected_all.shape[0]
        if bool(self.negative_config["random_wrong_connection"]) and target_count > 1:
            wrong_index = self._wrong_target_indices(
                source_index, target_index, source_count, target_count
            )
            negative_logits.append(
                self._indexed_consistency_logits(
                    source_projected, target_projected_all, wrong_index, relation
                )
            )
        if bool(self.negative_config["different_window_or_run"]):
            cross_index = self._cross_graph_target_indices(target_storage, target_index)
            if cross_index is not None:
                negative_logits.append(
                    self._indexed_consistency_logits(
                        source_projected, target_projected_all, cross_index, relation
                    )
                )
        if bool(self.negative_config["temporal_misalignment"]) and temporal_target_all is not None:
            negative_logits.append(
                self._indexed_consistency_logits(
                    source_projected, temporal_target_all, target_index, relation
                )
            )
        positive_loss = F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits)
        )
        if not negative_logits:
            return positive_loss
        combined_negative = torch.cat(negative_logits)
        negative_loss = F.binary_cross_entropy_with_logits(
            combined_negative, torch.zeros_like(combined_negative)
        )
        if bool(self.negative_config["balance_positive_negative"]):
            return 0.5 * (positive_loss + negative_loss)
        all_logits = torch.cat((positive_logits, combined_negative))
        all_labels = torch.cat((torch.ones_like(positive_logits), torch.zeros_like(combined_negative)))
        return F.binary_cross_entropy_with_logits(all_logits, all_labels)

    def forward(
        self,
        h_dict: Mapping[str, Tensor],
        data: HeteroData,
        reliability_dict: Mapping[str, Tensor],
        temporal_h_dict: Mapping[str, Tensor] | None,
        compute_auxiliary_loss: bool,
    ) -> tuple[dict[str, Tensor], dict[tuple[str, str, str], Tensor], Tensor]:
        aggregate = {node_type: torch.zeros_like(h_dict[node_type]) for node_type in NODE_TYPES}
        consistency_by_relation: dict[tuple[str, str, str], Tensor] = {}
        relation_losses: list[Tensor] = []
        for relation_id, edge_type in enumerate(EDGE_TYPES):
            if edge_type not in data.edge_types:
                continue
            source_type, _, target_type = edge_type
            edge_index = data[edge_type].edge_index
            if edge_index.numel() == 0:
                continue
            source_index, target_index = edge_index
            key = _relation_key(edge_type) if self.use_relation_types else "shared"
            consistency_source_all = self.source_projection[key](h_dict[source_type])
            consistency_target_all = self.target_projection[key](h_dict[target_type])
            consistency_source = consistency_source_all[source_index]
            consistency_target = consistency_target_all[target_index]
            embedding_index = relation_id if self.use_relation_types else 0
            relation = self.relation_embedding[embedding_index].expand(
                consistency_source.shape[0], -1
            )
            consistency_logits = self._consistency_logits(
                consistency_source, consistency_target, relation
            )
            consistency = torch.sigmoid(consistency_logits)
            consistency_by_relation[edge_type] = consistency

            relation_source_all = self.relation_projection[key](h_dict[source_type])
            relation_target_all = self.relation_projection[key](h_dict[target_type])
            relation_source = relation_source_all[source_index]
            relation_target = relation_target_all[target_index]
            base_features = torch.cat((relation_target, relation_source, relation), dim=-1)
            base_logits = F.leaky_relu(
                (base_features * self.attention_vector[key]).sum(dim=-1),
                negative_slope=self.negative_slope,
            )
            source_reliability = reliability_dict[source_type][source_index]
            if bool(self.switches["product_fusion"]):
                product = torch.ones_like(base_logits)
                if bool(self.switches["quality_attention"]):
                    product = product * source_reliability
                if bool(self.switches["consistency_attention"]):
                    product = product * consistency
                adjusted_logits = base_logits + torch.log(product + self.epsilon)
            else:
                adjusted_logits = base_logits
                if bool(self.switches["quality_attention"]):
                    adjusted_logits = adjusted_logits + self.quality_lambda[relation_id] * torch.log(
                        source_reliability + self.epsilon
                    )
                if bool(self.switches["consistency_attention"]):
                    adjusted_logits = adjusted_logits + self.consistency_gamma[relation_id] * consistency
            attention = softmax(adjusted_logits, target_index, num_nodes=h_dict[target_type].shape[0])
            aggregate[target_type].index_add_(
                0, target_index, attention[:, None] * relation_source
            )

            if compute_auxiliary_loss:
                temporal_target = None
                if temporal_h_dict is not None:
                    temporal_target = self.target_projection[key](temporal_h_dict[target_type])
                relation_losses.append(
                    self._consistency_loss(
                        consistency_logits,
                        consistency_source,
                        source_index,
                        h_dict[source_type].shape[0],
                        consistency_target_all,
                        target_index,
                        relation,
                        data[target_type],
                        temporal_target,
                    )
                )
        updated = {
            node_type: self.dropout(
                self.activation(self.self_projection[node_type](h_dict[node_type]) + aggregate[node_type])
            )
            for node_type in NODE_TYPES
        }
        zero = next(iter(h_dict.values())).new_zeros(())
        consistency_loss = torch.stack(relation_losses).mean() if relation_losses else zero
        return updated, consistency_by_relation, consistency_loss


class QCHGT(nn.Module):
    """Sections 4.5--4.8: quality, consistency, collaborative attention and classification."""

    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = int(config["hidden_dim"])
        self.switches = config["switches"]
        self.loss_weights = config["loss_weights"]
        input_dim = int(config["input_dim"])
        normalization = str(config["input_normalization"])
        if normalization not in {"layer_norm", "none"}:
            raise ValueError("qc_hgt.input_normalization must be layer_norm or none.")
        self.input_normalization = nn.ModuleDict(
            {
                node_type: nn.LayerNorm(input_dim) if normalization == "layer_norm" else nn.Identity()
                for node_type in NODE_TYPES
            }
        )
        self.input_projection = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, self.hidden_dim) for node_type in NODE_TYPES}
        )
        self.quality_estimator = QualityEstimator(config["quality"])
        self.layers = nn.ModuleList(
            [QCHGTLayer(config) for _ in range(int(config["num_layers"]))]
        )
        self.classifier = _mlp(
            input_dim=3 * self.hidden_dim,
            hidden_dims=config["classification"]["hidden_dims"],
            output_dim=num_classes,
            activation=str(config["classification"]["activation"]),
            dropout=float(config["classification"]["dropout"]),
        )

    @staticmethod
    def _batch_vector(data: HeteroData, node_type: str, device: torch.device) -> Tensor:
        storage = data[node_type]
        if hasattr(storage, "batch"):
            return storage.batch
        return torch.zeros(storage.x.shape[0], dtype=torch.long, device=device)

    def _reliability(
        self, data: HeteroData
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        device = data["sensor"].x.device
        descriptors = self.quality_estimator.features(
            data["sensor"].raw_window,
            data["sensor"].observed_mask,
            data["sensor"].timestamp_delay,
        )
        normalized_descriptors = self.quality_estimator.descriptor_normalization(descriptors)
        if bool(self.switches.get("fixed_observable_quality", False)):
            fixed = self.config["quality"]["fixed_observable"]
            weights = descriptors.new_tensor(fixed["weights"])
            scales = descriptors.new_tensor(fixed["scales"])
            if weights.numel() != descriptors.shape[1] or scales.numel() != descriptors.shape[1]:
                raise ValueError("Fixed observable weights/scales must match the six descriptors.")
            if (weights < 0.0).any() or (scales <= 0.0).any() or float(weights.sum()) <= 0.0:
                raise ValueError("Fixed observable weights/scales must be positive.")
            burden = descriptors.abs() / (descriptors.abs() + scales)
            sensor_reliability = (1.0 - (burden * weights).sum(dim=1) / weights.sum()).clamp(
                min=float(self.config["quality"]["epsilon"]), max=1.0
            )
        elif bool(self.switches["fixed_quality"]):
            if not hasattr(data["sensor"], "quality_target"):
                raise RuntimeError(
                    "Fixed Quality requires q_star and is forbidden for normal inference. "
                    "It may only be used as an explicitly labelled training/oracle diagnostic."
                )
            sensor_reliability = data["sensor"].quality_target.reshape(-1).detach()
        else:
            sensor_reliability = torch.sigmoid(
                self.quality_estimator.network(
                    normalized_descriptors
                )
            ).squeeze(-1)
        reliability = {
            "sensor": sensor_reliability,
            "actuator": torch.ones(data["actuator"].x.shape[0], device=device),
            "process_unit": torch.ones(data["process_unit"].x.shape[0], device=device),
        }
        quality_loss = sensor_reliability.new_zeros(())
        if hasattr(data["sensor"], "quality_target"):
            target = data["sensor"].quality_target.reshape(-1)
            quality_loss = F.mse_loss(sensor_reliability, target)
        return reliability, sensor_reliability, descriptors, quality_loss

    def forward(self, data: HeteroData) -> dict[str, Any]:
        if not isinstance(data, HeteroData):
            raise TypeError("QC-HGT requires PyG HeteroData; disable homogeneous_ablation.")
        reliability_dict, sensor_reliability, descriptors, quality_loss = self._reliability(data)
        h_dict = {
            node_type: self.input_projection[node_type](
                self.input_normalization[node_type](data[node_type].x)
            )
            for node_type in NODE_TYPES
        }
        temporal_h_dict: dict[str, Tensor] | None = None
        if all(hasattr(data[node_type], "temporal_negative_x") for node_type in NODE_TYPES):
            temporal_h_dict = {
                node_type: self.input_projection[node_type](
                    self.input_normalization[node_type](data[node_type].temporal_negative_x)
                )
                for node_type in NODE_TYPES
            }

        layer_losses: list[Tensor] = []
        consistency_scores: dict[tuple[str, str, str], Tensor] = {}
        for layer in self.layers:
            h_dict, consistency_scores, layer_loss = layer(
                h_dict,
                data,
                reliability_dict,
                temporal_h_dict,
                compute_auxiliary_loss=True,
            )
            layer_losses.append(layer_loss)
            if temporal_h_dict is not None:
                unit_reliability = {
                    node_type: torch.ones(
                        temporal_h_dict[node_type].shape[0],
                        device=temporal_h_dict[node_type].device,
                    )
                    for node_type in NODE_TYPES
                }
                temporal_h_dict, _, _ = layer(
                    temporal_h_dict,
                    data,
                    unit_reliability,
                    None,
                    compute_auxiliary_loss=False,
                )
        consistency_loss = torch.stack(layer_losses).mean() if layer_losses else quality_loss.new_zeros(())

        pooled = []
        for node_type in ("sensor", "process_unit", "actuator"):
            batch = self._batch_vector(data, node_type, h_dict[node_type].device)
            pooled.append(global_mean_pool(h_dict[node_type], batch))
        graph_embedding = torch.cat(pooled, dim=-1)
        logits = self.classifier(graph_embedding)
        probabilities = torch.softmax(logits, dim=-1)
        classification_loss = logits.new_zeros(())
        if hasattr(data, "y"):
            classification_loss = F.cross_entropy(logits, data.y.reshape(-1).long())

        quality_weight = (
            float(self.loss_weights["quality"])
            if bool(self.switches["quality_loss"])
            else 0.0
        )
        consistency_weight = (
            float(self.loss_weights["consistency"])
            if bool(self.switches["consistency_loss"])
            else 0.0
        )
        total_loss = (
            float(self.loss_weights["classification"]) * classification_loss
            + quality_weight * quality_loss
            + consistency_weight * consistency_loss
        )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "graph_embedding": graph_embedding,
            "reliability": sensor_reliability,
            "quality_features": descriptors,
            "consistency": consistency_scores,
            "loss": total_loss,
            "classification_loss": classification_loss,
            "quality_loss": quality_loss,
            "consistency_loss": consistency_loss,
        }


# Design note (paper Sec. 4.5--4.8): observable a_j predicts r_j and MSE L_q;
# relation-projected pairs predict c_ij and BCE L_c; both adjust relation logits
# before softmax; typed mean pooling yields graph logits and the joint loss.
