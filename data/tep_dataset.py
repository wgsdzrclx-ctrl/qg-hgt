"""Run-safe Tennessee Eastman Process (TEP) window datasets for PyG.

The module supports two configurable layouts:

``run_files``
    One CSV/TXT/DAT/NPY/NPZ/PT file per independent run.  A manifest is
    optional; otherwise fault/run identifiers are parsed from file names.

``table``
    Consolidated CSV or Parquet tables used by the 500-run extension, with
    ``faultNumber``, ``simulationRun``, ``sample`` and 52 process columns.

Runs are loaded and split before any window indices are built.  Each returned
PyG ``Data`` object contains 52 nodes and ``window.length`` temporal features.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, OrderedDict
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments
    raise ImportError(
        "tep_dataset.py requires PyTorch Geometric 2.x. Install a PyG build "
        "compatible with the installed PyTorch 2.x version."
    ) from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("Reading configs/data.yaml requires PyYAML.") from exc

from .corruptions import CorruptionResult, ProtocolCorruptor


@dataclass(frozen=True)
class RunRecord:
    """One independent TEP simulation sequence."""

    run_uid: str
    run_id: str
    source_split: str
    fault_id: int
    length: int
    origin: str
    values: Tensor | None = None  # [time, variables] for eager run-file data
    labels: Tensor | None = None  # [time] for eager run-file data
    parquet_path: str | None = None
    parquet_row_group: int | None = None
    feature_columns: tuple[str, ...] = ()
    sample_column: str | None = None
    normal_class: int = 0
    normal_before_onset: bool = False
    fault_onset_sample: int | None = None
    sample_index_base: int = 1


def load_data_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the YAML configuration."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"experiment", "dataset", "window", "corruptions", "quality", "protocols"}
    missing = required.difference(config)
    if missing:
        raise KeyError(f"Missing top-level config keys: {sorted(missing)}")
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError("experiment.seeds must be exactly [0, 1, 2, 3, 4].")
    variant = str(config["dataset"]["variant"])
    try:
        layout = config["dataset"]["variants"][variant]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset.variant: {variant}") from exc
    fault_ids = [int(value) for value in layout["fault_ids"]]
    if len(fault_ids) != len(set(fault_ids)):
        raise ValueError(f"variant={variant!r} contains duplicate fault IDs.")
    if int(layout["num_classes"]) != len(fault_ids):
        raise ValueError(
            f"variant={variant!r} declares num_classes={layout['num_classes']} "
            f"but has {len(fault_ids)} fault IDs."
        )
    config["_config_dir"] = str(config_path.parent)
    return config


def set_experiment_seed(seed: int, allowed_seeds: Sequence[int]) -> None:
    """Seed Python, NumPy, and PyTorch from the configured five-seed set."""
    seed = int(seed)
    if seed not in {int(value) for value in allowed_seeds}:
        raise ValueError(f"seed={seed} is not in configured seeds {list(allowed_seeds)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _parse_delimiter(first_line: str, configured: str) -> str | None:
    if configured != "auto":
        return None if configured == "whitespace" else configured
    return "," if "," in first_line else None


def _is_header(tokens: Sequence[str], configured: str | bool) -> bool:
    if isinstance(configured, bool):
        return configured
    if configured != "auto":
        raise ValueError("has_header must be true, false, or 'auto'.")
    try:
        for token in tokens:
            float(token)
        return False
    except ValueError:
        return True


def _ordered_feature_names(header: Sequence[str], config: Mapping[str, Any]) -> list[str]:
    explicit = config["feature_columns"]
    if explicit:
        return [str(name) for name in explicit]
    lower_to_original = {name.lower(): name for name in header}
    xmeas = int(config["xmeas_count"])
    xmv = int(config["xmv_count"])
    expected = [f"xmeas_{index}" for index in range(1, xmeas + 1)] + [
        f"xmv_{index}" for index in range(1, xmv + 1)
    ]
    if all(name in lower_to_original for name in expected):
        return [lower_to_original[name] for name in expected]
    expected_parentheses = [f"xmeas({index})" for index in range(1, xmeas + 1)] + [
        f"xmv({index})" for index in range(1, xmv + 1)
    ]
    if all(name in lower_to_original for name in expected_parentheses):
        return [lower_to_original[name] for name in expected_parentheses]
    raise ValueError("Could not infer the ordered XMEAS/XMV columns; set feature_columns in YAML.")


def _matrix_from_file(path: Path, layout: Mapping[str, Any], dataset_cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    suffix = path.suffix.lower()
    labels: np.ndarray | None = None
    if suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        matrix = archive[str(layout["npz_data_key"])]
        label_key = layout["npz_label_key"]
        if label_key and label_key in archive:
            labels = np.asarray(archive[label_key])
    elif suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, Mapping):
            matrix = torch.as_tensor(payload[str(layout["tensor_data_key"])]).cpu().numpy()
            label_key = layout["tensor_label_key"]
            if label_key and label_key in payload:
                labels = torch.as_tensor(payload[label_key]).cpu().numpy()
        else:
            matrix = torch.as_tensor(payload).cpu().numpy()
    else:
        with path.open("r", encoding=str(layout["encoding"])) as handle:
            first_line = handle.readline().strip()
        delimiter = _parse_delimiter(first_line, str(layout["delimiter"]))
        tokens = first_line.split(delimiter) if delimiter else first_line.split()
        has_header = _is_header(tokens, layout["has_header"])
        raw = np.genfromtxt(
            path,
            delimiter=delimiter,
            names=True if has_header else None,
            dtype=np.float64,
            encoding=str(layout["encoding"]),
        )
        if has_header:
            if raw.ndim == 0:
                raw = raw.reshape(1)
            header = list(raw.dtype.names or ())
            feature_names = _ordered_feature_names(header, dataset_cfg)
            matrix = np.column_stack([raw[name] for name in feature_names])
            label_name = layout["label_column"]
            if label_name and label_name in header:
                labels = np.asarray(raw[label_name])
        else:
            matrix = np.asarray(raw)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            label_index = layout["label_column_index"]
            if label_index is not None:
                labels = matrix[:, int(label_index)].copy()
                matrix = np.delete(matrix, int(label_index), axis=1)

    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D matrix, got {matrix.shape}")
    variable_count = int(dataset_cfg["num_variables"])
    if bool(layout["auto_transpose"]) and matrix.shape[0] == variable_count and matrix.shape[1] != variable_count:
        matrix = matrix.T
    if matrix.shape[1] != variable_count:
        raise ValueError(f"{path}: expected {variable_count} variables, got shape {matrix.shape}")
    return matrix.astype(np.float32, copy=False), None if labels is None else labels.astype(np.int64, copy=False)


def _scenario_labels(
    length: int,
    fault_id: int,
    source_split: str,
    labeling: Mapping[str, Any],
    samples: np.ndarray | None = None,
) -> np.ndarray:
    labels = np.full(length, int(fault_id), dtype=np.int64)
    if fault_id == int(labeling["normal_class"]):
        return labels
    if not bool(labeling["normal_before_onset"]):
        return labels
    onset = labeling["fault_onset_sample"][source_split]
    if onset is None:
        return labels
    normal = int(labeling["normal_class"])
    if samples is None:
        sample_base = int(labeling["sample_index_base"])
        samples = np.arange(length, dtype=np.int64) + sample_base
    labels[np.asarray(samples) < int(onset)] = normal
    return labels


def _match_identifier(path: Path, pattern: str, group: str) -> str | None:
    match = re.search(pattern, path.stem, flags=re.IGNORECASE)
    return match.group(group) if match else None


def _manifest_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_run_files(config: Mapping[str, Any], layout: Mapping[str, Any]) -> list[RunRecord]:
    dataset_cfg = config["dataset"]
    labeling = layout.get("labeling", dataset_cfg["labeling"])
    root = _resolve_path(config, dataset_cfg["root"])
    entries: list[tuple[str, Path, str | None, int | None]] = []
    manifest_value = layout["manifest"]
    if manifest_value:
        manifest = _resolve_path(config, manifest_value)
        for row in _manifest_rows(manifest):
            source_split = row[str(layout["manifest_columns"]["split"])]
            relative = Path(row[str(layout["manifest_columns"]["path"])])
            path = relative if relative.is_absolute() else root / relative
            run_id = row[str(layout["manifest_columns"]["run_id"])]
            fault_id = int(row[str(layout["manifest_columns"]["fault_id"])] )
            entries.append((source_split, path.resolve(), run_id, fault_id))
    else:
        extensions = {str(value).lower() for value in layout["extensions"]}
        for source_split in ("train", "test"):
            directory = root / str(layout[f"{source_split}_dir"])
            if not directory.exists():
                raise FileNotFoundError(f"Configured TEP directory does not exist: {directory}")
            paths = sorted(
                (path for path in directory.glob(str(layout["file_glob"])) if path.is_file() and path.suffix.lower() in extensions),
                key=lambda path: _natural_key(str(path)),
            )
            entries.extend((source_split, path, None, None) for path in paths)

    records: list[RunRecord] = []
    for source_split, path, manifest_run, manifest_fault in entries:
        values, file_labels = _matrix_from_file(path, layout, dataset_cfg)
        fault_text = _match_identifier(path, str(layout["fault_regex"]), "fault")
        fault_id = manifest_fault if manifest_fault is not None else (int(fault_text) if fault_text is not None else None)
        if fault_id is None and file_labels is not None:
            fault_id = int(np.max(file_labels))
        if fault_id is None:
            raise ValueError(f"Could not determine fault ID for {path}; use a manifest or fault_regex.")
        parsed_run = _match_identifier(path, str(layout["run_regex"]), "run")
        run_id = str(manifest_run or parsed_run or path.stem)
        labels = file_labels
        if labels is None:
            labels = _scenario_labels(len(values), fault_id, source_split, labeling)
        if len(labels) != len(values):
            raise ValueError(f"{path}: label length {len(labels)} != data length {len(values)}")
        origin = str(path.resolve())
        run_uid = f"{source_split}|fault={fault_id}|run={run_id}"
        records.append(
            RunRecord(
                run_uid=run_uid,
                run_id=run_id,
                source_split=source_split,
                fault_id=int(fault_id),
                length=len(values),
                origin=origin,
                values=torch.from_numpy(values),
                labels=torch.from_numpy(labels),
            )
        )

    for source_split in ("train", "test"):
        expected = layout[f"expected_{source_split}_runs"]
        actual = sum(record.source_split == source_split for record in records)
        if expected is not None and actual != int(expected):
            message = f"Expected {expected} {source_split} runs, found {actual}."
            if bool(layout["strict_counts"]):
                raise ValueError(message)
            print(f"[TEP warning] {message}")
    return records


def _read_table(path: Path, table_cfg: Mapping[str, Any]):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The extended table layout requires pandas (and pyarrow for Parquet).") from exc
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported extended table type: {path}. Convert RData files to CSV or Parquet first.")


def _statistic_constant(row_group: Any, column_index: int, name: str, path: Path) -> Any:
    statistics = row_group.column(column_index).statistics
    if statistics is None or not statistics.has_min_max or statistics.min != statistics.max:
        raise ValueError(
            f"{path}: every Parquet row group must contain exactly one {name}; "
            "run scripts/prepare_tep_extended.py first."
        )
    return statistics.min


def _display_run_id(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(value)


def _load_parquet_run_groups(
    path: Path,
    source_split: str,
    dataset_cfg: Mapping[str, Any],
    layout: Mapping[str, Any],
    labeling: Mapping[str, Any],
    seen_per_fault: dict[int, int],
) -> list[RunRecord]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Run-optimized extended Parquet loading requires pyarrow.") from exc

    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    fault_column = str(layout["fault_column"])
    run_column = str(layout["run_column"])
    sample_column = str(layout["sample_column"])
    feature_names = tuple(_ordered_feature_names(names, dataset_cfg))
    fault_index = names.index(fault_column)
    run_index = names.index(run_column)
    allowed_faults = {int(value) for value in layout.get("fault_ids", dataset_cfg["fault_ids"])}
    max_runs = layout["max_runs_per_fault"]
    records: list[RunRecord] = []

    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        fault_id = int(_statistic_constant(row_group, fault_index, fault_column, path))
        run_value = _statistic_constant(row_group, run_index, run_column, path)
        if fault_id not in allowed_faults:
            continue
        count = seen_per_fault.get(fault_id, 0)
        if max_runs is not None and count >= int(max_runs):
            continue
        seen_per_fault[fault_id] = count + 1
        run_id = _display_run_id(run_value)
        origin = f"{path.resolve()}#row_group={row_group_index}"
        run_uid = f"{source_split}|fault={fault_id}|run={run_id}"
        records.append(
            RunRecord(
                run_uid=run_uid,
                run_id=run_id,
                source_split=source_split,
                fault_id=fault_id,
                length=int(row_group.num_rows),
                origin=origin,
                parquet_path=str(path.resolve()),
                parquet_row_group=row_group_index,
                feature_columns=feature_names,
                sample_column=sample_column,
                normal_class=int(labeling["normal_class"]),
                normal_before_onset=bool(labeling["normal_before_onset"]),
                fault_onset_sample=labeling["fault_onset_sample"][source_split],
                sample_index_base=int(labeling["sample_index_base"]),
            )
        )
    return records


def _load_table_runs(config: Mapping[str, Any], layout: Mapping[str, Any]) -> list[RunRecord]:
    dataset_cfg = config["dataset"]
    labeling = layout.get("labeling", dataset_cfg["labeling"])
    root = _resolve_path(config, dataset_cfg["root"])
    records: list[RunRecord] = []
    max_runs = layout["max_runs_per_fault"]
    for source_split in ("train", "test"):
        seen_per_fault: dict[int, int] = {}
        for relative in layout["tables"][source_split]:
            path = (root / str(relative)).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Configured TEP table does not exist: {path}")
            if path.suffix.lower() == ".parquet" and bool(layout.get("parquet_row_group_per_run", False)):
                records.extend(
                    _load_parquet_run_groups(
                        path, source_split, dataset_cfg, layout, labeling, seen_per_fault
                    )
                )
                continue
            frame = _read_table(path, layout)
            fault_column = str(layout["fault_column"])
            run_column = str(layout["run_column"])
            sample_column = str(layout["sample_column"])
            feature_names = _ordered_feature_names(list(frame.columns), dataset_cfg)
            for (fault_value, run_value), group in frame.groupby([fault_column, run_column], sort=False):
                fault_id = int(fault_value)
                if fault_id not in {
                    int(value) for value in layout.get("fault_ids", dataset_cfg["fault_ids"])
                }:
                    continue
                count = seen_per_fault.get(fault_id, 0)
                if max_runs is not None and count >= int(max_runs):
                    continue
                seen_per_fault[fault_id] = count + 1
                group = group.sort_values(sample_column)
                values = group[feature_names].to_numpy(dtype=np.float32, copy=True)
                samples = group[sample_column].to_numpy(dtype=np.int64, copy=True)
                labels = _scenario_labels(len(values), fault_id, source_split, labeling, samples)
                run_id = str(run_value)
                origin = f"{path.resolve()}#{fault_id}:{run_id}"
                run_uid = f"{source_split}|fault={fault_id}|run={run_id}"
                records.append(
                    RunRecord(
                        run_uid=run_uid,
                        run_id=run_id,
                        source_split=source_split,
                        fault_id=fault_id,
                        length=len(values),
                        origin=origin,
                        values=torch.from_numpy(values),
                        labels=torch.from_numpy(labels),
                    )
                )
    expected = layout["expected_runs_per_fault"]
    if expected is not None and max_runs is None:
        for source_split in ("train", "test"):
            for fault_id in (
                int(value) for value in layout.get("fault_ids", dataset_cfg["fault_ids"])
            ):
                actual = sum(r.source_split == source_split and r.fault_id == fault_id for r in records)
                if actual != int(expected):
                    message = f"Expected {expected} runs for split={source_split}, fault={fault_id}; found {actual}."
                    if bool(layout["strict_counts"]):
                        raise ValueError(message)
                    print(f"[TEP warning] {message}")
    return records


def _load_runs(config: Mapping[str, Any]) -> list[RunRecord]:
    variant = str(config["dataset"]["variant"])
    try:
        layout = config["dataset"]["variants"][variant]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset.variant: {variant}") from exc
    layout_name = str(layout["layout"])
    if layout_name == "run_files":
        records = _load_run_files(config, layout)
    elif layout_name == "table":
        records = _load_table_runs(config, layout)
    else:
        raise ValueError(f"Unsupported layout: {layout_name}")

    logical_ids = [record.run_uid for record in records]
    duplicates = sorted(run_uid for run_uid, count in Counter(logical_ids).items() if count > 1)
    if duplicates:
        preview = duplicates[:5]
        raise ValueError(
            "Duplicate logical run IDs were loaded within a source namespace: "
            f"{preview}{' ...' if len(duplicates) > len(preview) else ''}"
        )
    return records


def _split_train_validation(
    records: Sequence[RunRecord], split_cfg: Mapping[str, Any], seed: int
) -> tuple[list[RunRecord], list[RunRecord]]:
    fraction = float(split_cfg["validation_fraction"])
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    generator = torch.Generator().manual_seed(int(seed))
    train: list[RunRecord] = []
    validation: list[RunRecord] = []
    if bool(split_cfg["stratify_by_fault"]):
        fault_ids = sorted({record.fault_id for record in records})
        for fault_id in fault_ids:
            group = [record for record in records if record.fault_id == fault_id]
            order = torch.randperm(len(group), generator=generator).tolist()
            if len(group) == 1:
                if split_cfg["singleton_policy"] == "validation":
                    validation.append(group[order[0]])
                else:
                    train.append(group[order[0]])
                continue
            n_validation = max(1, int(round(len(group) * fraction)))
            n_validation = min(n_validation, len(group) - 1)
            validation.extend(group[index] for index in order[:n_validation])
            train.extend(group[index] for index in order[n_validation:])
    else:
        order = torch.randperm(len(records), generator=generator).tolist()
        n_validation = max(1, int(round(len(records) * fraction)))
        n_validation = min(n_validation, len(records) - 1)
        validation = [records[index] for index in order[:n_validation]]
        train = [records[index] for index in order[n_validation:]]
    return train, validation


class TEPWindowDataset(Dataset[Data]):
    """A lazy window view over already split independent TEP runs."""

    def __init__(
        self,
        runs: Sequence[RunRecord],
        split: str,
        window_length: int,
        stride: int,
        nan_fill_value: float,
        corruptor: Callable[[Tensor, str, int], CorruptionResult] | None,
        expose_quality_target: bool,
        run_cache_size: int,
        shared_run_cache: OrderedDict[int, tuple[Tensor, Tensor]] | None = None,
        shared_parquet_files: dict[str, Any] | None = None,
    ) -> None:
        self.runs = list(runs)
        self.split = split
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.nan_fill_value = float(nan_fill_value)
        self.corruptor = corruptor
        self.expose_quality_target = bool(expose_quality_target)
        self.run_cache_size = int(run_cache_size)
        self._run_cache = shared_run_cache if shared_run_cache is not None else OrderedDict()
        self._parquet_files = shared_parquet_files if shared_parquet_files is not None else {}
        self.epoch = 0
        if self.window_length <= 0 or self.stride <= 0 or self.run_cache_size <= 0:
            raise ValueError("Window length, stride, and run cache size must be positive.")
        counts = [max(0, (run.length - self.window_length) // self.stride + 1) for run in self.runs]
        self._cumulative: list[int] = []
        total = 0
        for count in counts:
            total += count
            self._cumulative.append(total)

    def set_epoch(self, epoch: int) -> None:
        """Change stochastic training corruptions deterministically per epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_run_cache"] = OrderedDict()
        state["_parquet_files"] = {}
        return state

    def _run_tensors(self, run_index: int) -> tuple[Tensor, Tensor]:
        run = self.runs[run_index]
        if run.values is not None and run.labels is not None:
            return run.values, run.labels
        cached = self._run_cache.get(run_index)
        if cached is not None:
            self._run_cache.move_to_end(run_index)
            return cached
        if run.parquet_path is None or run.parquet_row_group is None or run.sample_column is None:
            raise RuntimeError(f"Run {run.run_uid} has neither eager tensors nor a Parquet source.")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Lazy extended Parquet loading requires pyarrow.") from exc
        parquet = self._parquet_files.get(run.parquet_path)
        if parquet is None:
            parquet = pq.ParquetFile(run.parquet_path)
            self._parquet_files[run.parquet_path] = parquet
        columns = [run.sample_column, *run.feature_columns]
        table = parquet.read_row_group(run.parquet_row_group, columns=columns)
        samples = table[run.sample_column].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        values_array = np.column_stack(
            [table[name].to_numpy(zero_copy_only=False) for name in run.feature_columns]
        ).astype(np.float32, copy=False)
        if len(values_array) != run.length:
            raise RuntimeError(f"{run.origin}: expected {run.length} rows, read {len(values_array)}")
        labels_array = _scenario_labels(
            run.length,
            run.fault_id,
            run.source_split,
            {
                "normal_class": run.normal_class,
                "normal_before_onset": run.normal_before_onset,
                "sample_index_base": run.sample_index_base,
                "fault_onset_sample": {run.source_split: run.fault_onset_sample},
            },
            samples,
        )
        tensors = (torch.from_numpy(values_array), torch.from_numpy(labels_array))
        self._run_cache[run_index] = tensors
        self._run_cache.move_to_end(run_index)
        while len(self._run_cache) > self.run_cache_size:
            self._run_cache.popitem(last=False)
        return tensors

    def __getitem__(self, index: int) -> Data:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        run_index = bisect_right(self._cumulative, index)
        preceding = 0 if run_index == 0 else self._cumulative[run_index - 1]
        local_window = index - preceding
        start = local_window * self.stride
        end = start + self.window_length
        run = self.runs[run_index]
        values, labels = self._run_tensors(run_index)
        clean = values[start:end].clone()
        sample_key = f"{run.run_uid}|start={start}|end={end}"
        if self.corruptor is None:
            observed = torch.isfinite(clean)
            q_star = torch.ones(clean.shape[1], dtype=clean.dtype)
            corrupted = clean
        else:
            output = self.corruptor(clean, sample_key, self.epoch)
            corrupted, observed, q_star = output.x, output.observed_mask, output.q_star
        model_x = torch.nan_to_num(corrupted, nan=self.nan_fill_value)
        data = Data(
            x=model_x.transpose(0, 1).contiguous(),
            observed_mask=observed.transpose(0, 1).contiguous(),
            y=labels[end - 1].reshape(1).long(),
            run_index=torch.tensor([run_index], dtype=torch.long),
            window_start=torch.tensor([start], dtype=torch.long),
            window_end=torch.tensor([end - 1], dtype=torch.long),
            fault_scenario=torch.tensor([run.fault_id], dtype=torch.long),
            run_uid=run.run_uid,
        )
        if self.expose_quality_target:
            data.q_star = q_star
        return data


def leakage_report(datasets: Mapping[str, TEPWindowDataset], fail_on_leakage: bool = True) -> dict[str, Any]:
    """Print and return logical-ID and physical-origin disjointness checks."""
    split_names = ("train", "val", "test")
    run_sets = {name: {run.run_uid for run in datasets[name].runs} for name in split_names}
    origin_sets = {name: {run.origin for run in datasets[name].runs} for name in split_names}
    overlaps: dict[str, list[str]] = {}
    origin_overlaps: dict[str, list[str]] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            key = f"{left}/{right}"
            overlaps[key] = sorted(run_sets[left].intersection(run_sets[right]))
            origin_overlaps[key] = sorted(origin_sets[left].intersection(origin_sets[right]))
    print("\n[TEP run-level leakage self-check]")
    for name in split_names:
        print(f"  {name:>5}: runs={len(run_sets[name]):5d}, windows={len(datasets[name]):9d}")
    for pair, values in overlaps.items():
        print(
            f"  {pair:>10} logical-ID overlap: {len(values)}; "
            f"physical-origin overlap: {len(origin_overlaps[pair])}"
        )
    passed = not any(overlaps.values()) and not any(origin_overlaps.values())
    print(f"  status: {'PASS' if passed else 'FAIL'} (split runs first, windowed second)\n")
    if fail_on_leakage and not passed:
        raise RuntimeError(
            f"Run leakage detected: logical={overlaps}, physical={origin_overlaps}"
        )
    return {
        "passed": passed,
        "overlaps": overlaps,
        "origin_overlaps": origin_overlaps,
        "run_counts": {key: len(value) for key, value in run_sets.items()},
    }


def build_tep_datasets(
    config_path: str | Path,
    seed: int,
    test_scenario: str | None = None,
    print_leakage_report: bool = True,
) -> dict[str, TEPWindowDataset]:
    """Build train/validation/test datasets with run-first splitting."""
    config = load_data_config(config_path)
    return build_tep_datasets_from_config(
        config,
        seed=seed,
        test_scenario=test_scenario,
        print_leakage_report=print_leakage_report,
    )


def build_tep_datasets_from_config(
    config: Mapping[str, Any],
    seed: int,
    test_scenario: str | None = None,
    print_leakage_report: bool = True,
) -> dict[str, TEPWindowDataset]:
    """Build datasets from an already loaded config, useful for variant checks."""
    set_experiment_seed(seed, config["experiment"]["seeds"])
    all_runs = _load_runs(config)
    source_train = [run for run in all_runs if run.source_split == "train"]
    source_test = [run for run in all_runs if run.source_split == "test"]
    variant = str(config["dataset"]["variant"])
    layout = config["dataset"]["variants"][variant]
    split_config = layout.get("split", config["dataset"]["split"])
    split_seed = int(split_config["seed"])
    train_runs, validation_runs = _split_train_validation(source_train, split_config, split_seed)
    if not train_runs or not validation_runs or not source_test:
        raise ValueError(
            f"variant={variant!r} produced an empty run split: "
            f"train={len(train_runs)}, val={len(validation_runs)}, test={len(source_test)}"
        )
    run_splits = {"train": train_runs, "val": validation_runs, "test": source_test}
    active_protocol = str(config["protocols"]["active"])
    if test_scenario is None:
        test_scenario = config["evaluation"]["test_scenario_by_protocol"][active_protocol]
    quality_visibility = config["quality"]["attach_target"]
    datasets: dict[str, TEPWindowDataset] = {}
    for split, runs in run_splits.items():
        scenario = test_scenario if split == "test" else None
        corruptor = ProtocolCorruptor(config, split, seed, scenario)
        datasets[split] = TEPWindowDataset(
            runs=runs,
            split=split,
            window_length=int(config["window"]["length"]),
            stride=int(config["window"]["stride"]),
            nan_fill_value=float(config["dataset"]["nan_fill_value"]),
            corruptor=corruptor,
            expose_quality_target=bool(quality_visibility[split]),
            run_cache_size=int(config["dataset"]["run_cache_size"]),
        )
    if print_leakage_report:
        leakage_report(datasets, fail_on_leakage=bool(split_config["fail_on_leakage"]))
    return datasets


def build_tep_evaluation_grid(
    config_path: str | Path,
    seed: int,
) -> dict[str, TEPWindowDataset]:
    """Build clean and fixed single-corruption test sets for robustness curves."""
    config = load_data_config(config_path)
    return build_tep_evaluation_grid_from_config(config, seed)


def build_tep_evaluation_grid_from_config(
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, TEPWindowDataset]:
    """Build the paired fixed-strength grid from an in-memory configuration."""
    set_experiment_seed(seed, config["experiment"]["seeds"])
    test_runs = [run for run in _load_runs(config) if run.source_split == "test"]
    grid = config["evaluation"]["fixed_strength_grid"]
    shared_run_cache: OrderedDict[int, tuple[Tensor, Tensor]] = OrderedDict()
    shared_parquet_files: dict[str, Any] = {}
    common = {
        "runs": test_runs,
        "split": "test",
        "window_length": int(config["window"]["length"]),
        "stride": int(config["window"]["stride"]),
        "nan_fill_value": float(config["dataset"]["nan_fill_value"]),
        "expose_quality_target": bool(config["quality"]["attach_target"]["test"]),
        "run_cache_size": int(config["dataset"]["run_cache_size"]),
        "shared_run_cache": shared_run_cache,
        "shared_parquet_files": shared_parquet_files,
    }
    datasets = {
        "clean": TEPWindowDataset(
            corruptor=ProtocolCorruptor(
                config, "test", seed, rule={"mode": "clean"}
            ),
            **common,
        )
    }
    for corruption_name, configured_levels in grid["levels"].items():
        allowed_levels = config["corruptions"][corruption_name]["levels"]
        for strength in configured_levels:
            if strength not in allowed_levels:
                raise ValueError(
                    f"evaluation.fixed_strength_grid contains unconfigured strength "
                    f"{corruption_name}={strength}"
                )
            name = f"{corruption_name}={strength}"
            rule = {
                "mode": "fixed",
                "combination": [corruption_name],
                "strengths": {corruption_name: strength},
                "variable_selection": grid["variable_selection"],
            }
            datasets[name] = TEPWindowDataset(
                corruptor=ProtocolCorruptor(config, "test", seed, rule=rule),
                **common,
            )
    return datasets


def build_tep_protocol_test_sets_from_config(
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, TEPWindowDataset]:
    """Build paired single-strength sets plus configured protocol test groups."""
    datasets = build_tep_evaluation_grid_from_config(config, seed)
    test_runs = next(iter(datasets.values())).runs
    common = {
        "runs": test_runs,
        "split": "test",
        "window_length": int(config["window"]["length"]),
        "stride": int(config["window"]["stride"]),
        "nan_fill_value": float(config["dataset"]["nan_fill_value"]),
        "expose_quality_target": bool(config["quality"]["attach_target"]["test"]),
        "run_cache_size": int(config["dataset"]["run_cache_size"]),
        "shared_run_cache": next(iter(datasets.values()))._run_cache,
        "shared_parquet_files": next(iter(datasets.values()))._parquet_files,
    }
    active = str(config["protocols"]["active"])
    for scenario, rule in config["protocols"][active]["test_scenarios"].items():
        datasets[f"protocol_group={scenario}"] = TEPWindowDataset(
            corruptor=ProtocolCorruptor(config, "test", seed, rule=rule),
            **common,
        )
    return datasets


# Design note (paper Sec. 4.1-4.2, 5.1): each sample is X_t=[x_(t-W+1),...,x_t]
# and y is the label at t. Runs are partitioned before lazy window indexing;
# fixed-strength and protocol-group datasets implement Sec. 5.2 and 5.4.
