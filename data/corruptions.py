"""Controlled node-wise corruptions for TEP time windows.

All public functions operate on a ``[time, variables]`` floating-point tensor.
They return :class:`CorruptionResult`.  For compatibility with a two-value API,
the result can be unpacked as ``corrupted_x, q_star = result``; the observation
mask remains available as ``result.observed_mask``.

The injected strengths are used only to construct the supervision target
``q_star``.  They are deliberately not encoded in ``x`` or the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


_COMPONENTS = ("noise", "missing", "drift", "time_shift")


@dataclass(frozen=True)
class CorruptionResult:
    """A corrupted window and its supervision-only quality information."""

    x: Tensor
    q_star: Tensor
    observed_mask: Tensor
    components: Mapping[str, Tensor]

    def __iter__(self):
        """Allow ``x_tilde, q_star = result`` without hiding the mask attribute."""
        yield self.x
        yield self.q_star


def _as_window(x: Tensor) -> Tensor:
    x = torch.as_tensor(x)
    if x.ndim != 2:
        raise ValueError(f"Expected [time, variables], got shape {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError("Corruptions require a floating-point tensor.")
    return x


def _variable_indices(variables: Sequence[int] | Tensor, width: int) -> list[int]:
    indices = [int(v) for v in torch.as_tensor(variables).flatten().tolist()]
    if not indices:
        raise ValueError("At least one target variable must be supplied.")
    if len(indices) != len(set(indices)):
        raise ValueError("Target variable indices must be unique.")
    if min(indices) < 0 or max(indices) >= width:
        raise IndexError(f"Variable indices must be within [0, {width - 1}].")
    return indices


def _quality_settings(config: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    beta_cfg = config["beta"]
    norm_cfg = config["normalization"]
    beta = {
        "noise": float(beta_cfg["gaussian_noise"]),
        "missing": float(beta_cfg["missing"]),
        "drift": float(beta_cfg["drift"]),
        "time_shift": float(beta_cfg["time_shift"]),
    }
    normalization = {
        "noise": float(norm_cfg["gaussian_sigma"]),
        "missing": float(norm_cfg["missing_rate"]),
        "drift": float(norm_cfg["drift_slope"]),
        "time_shift": float(norm_cfg["time_shift"]),
    }
    if any(value < 0.0 for value in beta.values()):
        raise ValueError("Quality beta coefficients must be non-negative.")
    if any(value <= 0.0 for value in normalization.values()):
        raise ValueError("Quality normalization constants must be positive.")
    return beta, normalization


def _empty_components(x: Tensor) -> dict[str, Tensor]:
    return {name: torch.zeros(x.shape[1], dtype=x.dtype, device=x.device) for name in _COMPONENTS}


def _quality_score(components: Mapping[str, Tensor], config: Mapping[str, Any]) -> Tensor:
    beta, _ = _quality_settings(config)
    penalty = sum(beta[name] * components[name] for name in _COMPONENTS)
    clamp = config["clamp"]
    return (1.0 - penalty).clamp(float(clamp[0]), float(clamp[1]))


def _normalised(value: float, denominator: float) -> float:
    return min(abs(float(value)) / denominator, 1.0)


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _ensure_generator(generator: torch.Generator | None, x: Tensor) -> torch.Generator:
    if generator is None:
        raise ValueError(
            "A seeded torch.Generator is required. Use ProtocolCorruptor or pass "
            "torch.Generator().manual_seed(seed) explicitly."
        )
    return generator


def _apply_sequence(
    x: Tensor,
    operations: Sequence[Mapping[str, Any]],
    quality_config: Mapping[str, Any],
    generator: torch.Generator | None,
) -> CorruptionResult:
    """Apply operations and evaluate the paper's additive quality formula once."""
    source = _as_window(x)
    result = source.clone()
    observed = torch.isfinite(result)
    components = _empty_components(result)
    _, normalization = _quality_settings(quality_config)
    rng = _ensure_generator(generator, result)
    time_steps, width = result.shape

    for operation in operations:
        name = str(operation["type"])
        variables = _variable_indices(operation["variables"], width)
        var_tensor = torch.tensor(variables, dtype=torch.long, device=result.device)

        if name == "gaussian_noise":
            sigma = float(operation["sigma"])
            if sigma < 0.0:
                raise ValueError("sigma must be non-negative.")
            noise = torch.randn(
                (time_steps, len(variables)),
                dtype=result.dtype,
                device=result.device,
                generator=rng,
            )
            current = result[:, var_tensor]
            result[:, var_tensor] = torch.where(observed[:, var_tensor], current + sigma * noise, current)
            burden = _normalised(sigma, normalization["noise"])
            components["noise"][var_tensor] = torch.maximum(
                components["noise"][var_tensor],
                torch.full((len(variables),), burden, dtype=result.dtype, device=result.device),
            )

        elif name == "drift":
            slope = float(operation["slope"])
            origin = operation["time_origin"]
            if origin == "window_start":
                t0 = 0.0
            elif origin == "window_center":
                t0 = (time_steps - 1) / 2.0
            else:
                t0 = float(origin)
            timeline = torch.arange(time_steps, dtype=result.dtype, device=result.device) - t0
            current = result[:, var_tensor]
            drifted = current + timeline[:, None] * slope
            result[:, var_tensor] = torch.where(observed[:, var_tensor], drifted, current)
            burden = _normalised(slope, normalization["drift"])
            components["drift"][var_tensor] = torch.maximum(
                components["drift"][var_tensor],
                torch.full((len(variables),), burden, dtype=result.dtype, device=result.device),
            )

        elif name == "random_missing":
            rate = float(operation["rate"])
            if not 0.0 <= rate <= 1.0:
                raise ValueError("Missing rate must lie in [0, 1].")
            count = int(round(time_steps * rate))
            fill_value = operation["fill_value"]
            for variable in variables:
                positions = torch.randperm(time_steps, generator=rng, device=result.device)[:count]
                observed[positions, variable] = False
                result[positions, variable] = fill_value

        elif name == "continuous_missing":
            length = int(operation["length"])
            if not 0 <= length <= time_steps:
                raise ValueError(f"Continuous missing length must lie in [0, {time_steps}].")
            fill_value = operation["fill_value"]
            max_start = time_steps - length
            for variable in variables:
                start = int(torch.randint(max_start + 1, (1,), generator=rng).item()) if length else 0
                observed[start : start + length, variable] = False
                result[start : start + length, variable] = fill_value

        elif name == "time_shift":
            delta = int(operation["delta"])
            padding = str(operation["padding"])
            indices = torch.arange(time_steps, device=result.device) - delta
            valid = (indices >= 0) & (indices < time_steps)
            clipped = indices.clamp(0, time_steps - 1)
            shifted_values = result[:, var_tensor].clone()[clipped]
            shifted_mask = observed[:, var_tensor].clone()[clipped]
            if padding in {"zero", "nan"}:
                fill = 0.0 if padding == "zero" else float("nan")
                shifted_values[~valid, :] = fill
                shifted_mask[~valid, :] = False
            elif padding != "edge":
                raise ValueError("Time-shift padding must be one of: edge, zero, nan.")
            result[:, var_tensor] = shifted_values
            observed[:, var_tensor] = shifted_mask
            burden = _normalised(delta, normalization["time_shift"])
            components["time_shift"][var_tensor] = torch.maximum(
                components["time_shift"][var_tensor],
                torch.full((len(variables),), burden, dtype=result.dtype, device=result.device),
            )
        else:
            raise KeyError(f"Unknown corruption type: {name}")

    missing_fraction = (~observed).to(result.dtype).mean(dim=0)
    components["missing"] = (missing_fraction / normalization["missing"]).clamp(0.0, 1.0)
    q_star = _quality_score(components, quality_config)
    return CorruptionResult(result, q_star, observed, components)


def gaussian_noise(
    x: Tensor,
    variables: Sequence[int] | Tensor,
    sigma: float,
    quality_config: Mapping[str, Any],
    generator: torch.Generator | None = None,
) -> CorruptionResult:
    """Apply ``x_tilde = x + N(0, sigma^2)`` to selected variables."""
    return _apply_sequence(
        x, [{"type": "gaussian_noise", "variables": variables, "sigma": sigma}], quality_config, generator
    )


def sensor_drift(
    x: Tensor,
    variables: Sequence[int] | Tensor,
    slope: float,
    time_origin: str | float,
    quality_config: Mapping[str, Any],
    generator: torch.Generator | None = None,
) -> CorruptionResult:
    """Apply ``x_tilde(t) = x(t) + slope * (t - t0)``."""
    return _apply_sequence(
        x,
        [{"type": "drift", "variables": variables, "slope": slope, "time_origin": time_origin}],
        quality_config,
        generator,
    )


def random_missing(
    x: Tensor,
    variables: Sequence[int] | Tensor,
    rate: float,
    fill_value: float,
    quality_config: Mapping[str, Any],
    generator: torch.Generator | None = None,
) -> CorruptionResult:
    """Mask an exact, randomly selected fraction of points per target variable."""
    return _apply_sequence(
        x,
        [{"type": "random_missing", "variables": variables, "rate": rate, "fill_value": fill_value}],
        quality_config,
        generator,
    )


def continuous_missing(
    x: Tensor,
    variables: Sequence[int] | Tensor,
    length: int,
    fill_value: float,
    quality_config: Mapping[str, Any],
    generator: torch.Generator | None = None,
) -> CorruptionResult:
    """Mask one random contiguous interval per target variable."""
    return _apply_sequence(
        x,
        [{"type": "continuous_missing", "variables": variables, "length": length, "fill_value": fill_value}],
        quality_config,
        generator,
    )


def time_shift(
    x: Tensor,
    variables: Sequence[int] | Tensor,
    delta: int,
    padding: str,
    quality_config: Mapping[str, Any],
    generator: torch.Generator | None = None,
) -> CorruptionResult:
    """Apply ``x_tilde(t) = x(t-delta)`` without circular wrap-around."""
    return _apply_sequence(
        x,
        [{"type": "time_shift", "variables": variables, "delta": delta, "padding": padding}],
        quality_config,
        generator,
    )


def apply_corruptions(
    x: Tensor,
    operations: Sequence[Mapping[str, Any]],
    quality_config: Mapping[str, Any],
    generator: torch.Generator | None = None,
) -> CorruptionResult:
    """Compose corruptions and compute one additive quality target per variable."""
    return _apply_sequence(x, operations, quality_config, generator)


class ProtocolCorruptor:
    """Deterministic per-window corruption sampler configured entirely from YAML."""

    def __init__(
        self,
        config: Mapping[str, Any],
        split: str,
        seed: int,
        scenario: str | None = None,
        rule: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.seed = int(seed)
        if scenario is not None and rule is not None:
            raise ValueError("Pass either a named scenario or an explicit rule, not both.")
        active = str(config["protocols"]["active"])
        protocol = config["protocols"][active]
        if rule is not None:
            self.rule = rule
        elif split == "test" and scenario is not None:
            try:
                self.rule = protocol["test_scenarios"][scenario]
            except KeyError as exc:
                raise KeyError(f"Unknown test scenario '{scenario}' for {active}.") from exc
        else:
            self.rule = protocol["splits"][split]

    def _seed_for(self, sample_key: str, epoch: int) -> int:
        payload = f"{self.seed}|{self.split}|{epoch}|{sample_key}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

    @staticmethod
    def _choice(items: Sequence[Any], generator: torch.Generator) -> Any:
        if not items:
            raise ValueError("Cannot sample from an empty configured list.")
        index = int(torch.randint(len(items), (1,), generator=generator).item())
        return items[index]

    def _variables(self, width: int, generator: torch.Generator) -> list[int]:
        selection = self.rule["variable_selection"]
        included = selection["include"] or list(range(width))
        excluded = {int(v) for v in selection["exclude"]}
        candidates = [int(v) for v in included if int(v) not in excluded]
        low, high = (int(v) for v in selection["count"])
        high = min(high, len(candidates))
        if low < 1 or low > high:
            raise ValueError("Invalid variable_selection.count for available variables.")
        count = int(torch.randint(low, high + 1, (1,), generator=generator).item())
        order = torch.randperm(len(candidates), generator=generator)[:count].tolist()
        return [candidates[index] for index in order]

    def __call__(self, x: Tensor, sample_key: str, epoch: int = 0) -> CorruptionResult:
        x = _as_window(x)
        generator = _make_generator(self._seed_for(sample_key, epoch), x.device)
        mode = str(self.rule["mode"])
        if mode == "clean":
            return apply_corruptions(x, [], self.config["quality"], generator)

        if mode == "sample":
            probability = float(self.rule["probability"])
            draw = float(torch.rand((), generator=generator).item())
            if draw > probability:
                return apply_corruptions(x, [], self.config["quality"], generator)
            combination = self._choice(self.rule["combinations"], generator)
            strengths = {
                name: self._choice(self.rule["levels"][name], generator) for name in combination
            }
        elif mode == "fixed":
            combination = list(self.rule["combination"])
            strengths = self.rule["strengths"]
        else:
            raise ValueError(f"Corruption mode must be clean, sample, or fixed; got {mode!r}.")

        variables = self._variables(x.shape[1], generator)
        operations: list[dict[str, Any]] = []
        for name in combination:
            strength = strengths[name]
            operation: dict[str, Any] = {"type": name, "variables": variables}
            if name == "gaussian_noise":
                operation["sigma"] = float(strength)
            elif name == "drift":
                operation.update(slope=float(strength), time_origin=self.config["corruptions"][name]["time_origin"])
            elif name == "random_missing":
                operation.update(rate=float(strength), fill_value=self.config["corruptions"][name]["fill_value"])
            elif name == "continuous_missing":
                operation.update(length=int(strength), fill_value=self.config["corruptions"][name]["fill_value"])
            elif name == "time_shift":
                operation.update(delta=int(strength), padding=self.config["corruptions"][name]["padding"])
            else:
                raise KeyError(f"Unknown corruption type: {name}")
            operations.append(operation)
        return apply_corruptions(x, operations, self.config["quality"], generator)


# Design note (paper Sec. 4.4): the five operators implement the stated
# degradation equations.  ``components`` stores normalized sigma/m/a/delta,
# and q* is computed once as 1-sum(beta_k * component_k), then clamped.
