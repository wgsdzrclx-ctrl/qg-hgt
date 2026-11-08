"""Aggregate Stage-5 JSON results into paper tables and sensitivity data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_results(root: Path) -> list[dict[str, Any]]:
    """Load only complete Stage-5 result documents, recursively."""
    documents: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("schema_version") == "stage5.v1" and bool(value.get("complete")):
            value["_path"] = str(path)
            documents.append(value)
    return documents


def _scenario_values(document: Mapping[str, Any], key: str, missing_names: Iterable[str]) -> list[float]:
    evaluations = document["evaluations"]
    if key in evaluations:
        return [float(evaluations[key]["macro_f1"])]
    names = list(missing_names) if key == "missing" else [key]
    values = []
    for name, metrics in evaluations.items():
        if any(name.startswith(f"{candidate}=") for candidate in names):
            values.append(float(metrics["macro_f1"]))
    return values


def _cell(values: list[float], ddof: int) -> dict[str, Any]:
    if not values:
        return {"values": [], "mean": None, "std": None, "text": "--"}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if ddof == 1 and len(values) > 1 else statistics.pstdev(values)
    return {"values": values, "mean": mean, "std": std, "text": f"{mean:.4f} ± {std:.4f}"}


def _main_rows(
    documents: list[dict[str, Any]],
    protocol: str,
    models: list[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    columns = config["summary"]["main_columns"]
    missing = config["summary"]["missing_corruptions"]
    ddof = int(config["summary"]["std_ddof"])
    rows = []
    for model in models:
        selected = [
            doc for doc in documents
            if doc["protocol"] == protocol and doc["experiment_type"] == "main" and doc["model"] == model
        ]
        row: dict[str, Any] = {"model": model, "seeds": sorted(int(doc["seed"]) for doc in selected)}
        for display, key in columns.items():
            per_seed = []
            for document in selected:
                values = _scenario_values(document, str(key), missing)
                if values:
                    per_seed.append(statistics.fmean(values))
            row[display] = _cell(per_seed, ddof)
        rows.append(row)
    return rows


def _combined_main_rows(
    documents: list[dict[str, Any]],
    models: list[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the paper main table from the configured source protocol per column."""
    columns = config["summary"]["main_columns"]
    sources = config["summary"]["combined_main_sources"]
    missing = config["summary"]["missing_corruptions"]
    ddof = int(config["summary"]["std_ddof"])
    if set(columns) != set(sources):
        raise ValueError("summary.combined_main_sources must define every main table column.")
    rows = []
    for model in models:
        row: dict[str, Any] = {"model": model, "seeds": []}
        used_seeds: set[int] = set()
        for display, key in columns.items():
            protocol = str(sources[display])
            selected = [
                doc for doc in documents
                if doc["protocol"] == protocol
                and doc["experiment_type"] == "main"
                and doc["model"] == model
            ]
            per_seed = []
            for document in selected:
                values = _scenario_values(document, str(key), missing)
                if values:
                    per_seed.append(statistics.fmean(values))
                    used_seeds.add(int(document["seed"]))
            row[display] = _cell(per_seed, ddof)
        row["seeds"] = sorted(used_seeds)
        rows.append(row)
    return rows


def _latex(rows: list[dict[str, Any]], columns: list[str], caption: str, label: str) -> str:
    column_spec = "l" + "c" * len(columns)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        "Model & " + " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        model = str(row["model"]).replace("_", "\\_")
        cells = [str(row[column]["text"]).replace("±", "$\\pm$") for column in columns]
        lines.append(model + " & " + " & ".join(cells) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}", ""))
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seeds", *columns])
        for row in rows:
            writer.writerow([row["model"], ",".join(map(str, row["seeds"])), *[row[c]["text"] for c in columns]])


def _ablation_rows(
    documents: list[dict[str, Any]], protocol: str, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    ddof = int(config["summary"]["std_ddof"])
    rows = []
    for display, model in config["summary"]["ablation_rows"]:
        selected = [
            doc for doc in documents
            if doc["protocol"] == protocol and doc["model"] == model
            and doc["experiment_type"] in {"main", "ablation"}
        ]
        clean = [float(doc["evaluations"]["clean"]["macro_f1"]) for doc in selected if "clean" in doc["evaluations"]]
        seen = [
            float(doc["evaluations"]["protocol_group=known"]["macro_f1"])
            for doc in selected if "protocol_group=known" in doc["evaluations"]
        ]
        unseen = [
            float(doc["evaluations"]["protocol_group=unseen_combination"]["macro_f1"])
            for doc in selected if "protocol_group=unseen_combination" in doc["evaluations"]
        ]
        rows.append(
            {
                "model": str(display),
                "seeds": sorted(int(doc["seed"]) for doc in selected),
                "Clean": _cell(clean, ddof),
                "SeenDegradation": _cell(seen, ddof),
                "UnseenCombination": _cell(unseen, ddof),
            }
        )
    return rows


def _protocol_group_rows(
    documents: list[dict[str, Any]], protocol: str, models: list[str], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    ddof = int(config["summary"]["std_ddof"])
    rows = []
    for model in models:
        selected = [
            doc for doc in documents
            if doc["protocol"] == protocol and doc["experiment_type"] == "main" and doc["model"] == model
        ]
        row: dict[str, Any] = {"model": model, "seeds": sorted(int(doc["seed"]) for doc in selected)}
        for display, scenario in (
            ("Known", "protocol_group=known"),
            ("UnseenCombination", "protocol_group=unseen_combination"),
        ):
            values = [
                float(doc["evaluations"][scenario]["macro_f1"])
                for doc in selected if scenario in doc["evaluations"]
            ]
            row[display] = _cell(values, ddof)
        rows.append(row)
    return rows


def _sensitivity_rows(documents: list[dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    ddof = int(config["summary"]["std_ddof"])
    scenario = str(config["sensitivity"]["evaluation_scenario"])
    rows = []
    for parameter in ("window_length", "lambda_q", "lambda_c"):
        for value in config["sensitivity"][parameter]:
            selected = [
                doc for doc in documents
                if doc["experiment_type"] == "sensitivity"
                and doc.get("sensitivity_parameter") == parameter
                and float(doc.get("sensitivity_value")) == float(value)
            ]
            scores = [
                float(doc["evaluations"][scenario]["macro_f1"])
                for doc in selected if scenario in doc["evaluations"]
            ]
            cell = _cell(scores, ddof)
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "seeds": sorted(int(doc["seed"]) for doc in selected),
                    "macro_f1_mean": cell["mean"],
                    "macro_f1_std": cell["std"],
                    "mean_plus_minus_std": cell["text"],
                }
            )
    return rows


def summarize(config_path: Path) -> Path:
    config = _load_yaml(config_path)
    results_root = _resolve(config_path.parent, str(config["output"]["results_directory"]))
    output = results_root / str(config["output"]["summary_subdirectory"])
    output.mkdir(parents=True, exist_ok=True)
    documents = load_results(results_root)
    columns = list(config["summary"]["main_columns"])
    main_models = [str(v) for v in config["models"]["main"]]
    for protocol in config["protocols"]:
        protocol_models = [
            str(v)
            for v in config["protocols"][protocol].get("main_models", main_models)
        ]
        rows = _main_rows(documents, protocol, protocol_models, config)
        _write_csv(output / f"main_results_{protocol}.csv", rows, columns)
        with (output / f"main_results_{protocol}.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=int(config["output"]["json_indent"]))
        (output / f"main_results_{protocol}.tex").write_text(
            _latex(rows, columns, f"Main results for {protocol}", f"tab:main-{protocol}"), encoding="utf-8"
        )
        if protocol in {str(v) for v in config["summary"]["ablation_protocols"]}:
            ablations = _ablation_rows(documents, protocol, config)
            ablation_columns = ["Clean", "SeenDegradation", "UnseenCombination"]
            _write_csv(output / f"ablation_{protocol}.csv", ablations, ablation_columns)
            (output / f"ablation_{protocol}.tex").write_text(
                _latex(ablations, ablation_columns, f"Ablation results for {protocol}", f"tab:ablation-{protocol}"),
                encoding="utf-8",
            )
        if protocol in {str(v) for v in config["summary"]["group_protocols"]}:
            groups = _protocol_group_rows(documents, protocol, protocol_models, config)
            _write_csv(output / f"protocol_groups_{protocol}.csv", groups, ["Known", "UnseenCombination"])
            (output / f"protocol_groups_{protocol}.tex").write_text(
                _latex(
                    groups,
                    ["Known", "UnseenCombination"],
                    f"Known and unseen degradation groups for {protocol}",
                    f"tab:groups-{protocol}",
                ),
                encoding="utf-8",
            )

    combined_models = [str(v) for v in config["summary"]["combined_main_models"]]
    combined = _combined_main_rows(documents, combined_models, config)
    _write_csv(output / "main_results.csv", combined, columns)
    with (output / "main_results.json").open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, ensure_ascii=False, indent=int(config["output"]["json_indent"]))
    (output / "main_results.tex").write_text(
        _latex(
            combined,
            columns,
            "Main results under clean, single-degradation, and mixed-degradation protocols",
            "tab:main-results",
        ),
        encoding="utf-8",
    )

    sensitivity = _sensitivity_rows(documents, config)
    with (output / "sensitivity.json").open("w", encoding="utf-8") as handle:
        json.dump(sensitivity, handle, ensure_ascii=False, indent=int(config["output"]["json_indent"]))
    with (output / "sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity[0]) if sensitivity else ["parameter", "value"])
        writer.writeheader()
        writer.writerows(sensitivity)
    manifest = {
        "schema_version": "stage5.summary.v1",
        "source_result_count": len(documents),
        "output_directory": str(output),
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=int(config["output"]["json_indent"]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage5.yaml")
    arguments = parser.parse_args()
    destination = summarize(arguments.config.expanduser().resolve())
    print(f"Stage-5 summaries written to: {destination}")


if __name__ == "__main__":
    main()


# Design note (paper Sec. 5.5--5.7): seed-level JSON is aggregated without
# pooling predictions across seeds, preserving mean +/- sample-standard-deviation reporting.
