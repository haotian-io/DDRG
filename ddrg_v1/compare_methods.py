import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddrg_v1 import utils as ddrg
from ddrg_v1.summarize_results import load_rows, markdown_table


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid input spec {spec!r}; expected label=path.")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid input spec {spec!r}; label is empty.")
    return label, Path(raw_path).resolve()


def resolve_input_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("*.jsonl") if candidate.is_file())
    return [path]


def load_labeled_rows(specs: list[str]) -> dict[str, list[dict[str, Any]]]:
    labeled_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        label, path = parse_input_spec(spec)
        paths = resolve_input_paths(path)
        labeled_rows[label] = load_rows(paths)
    return labeled_rows


def probe_count(row: dict[str, Any]) -> int:
    return len(row.get("ddrg_v1", {}).get("probe_results", []))


def graph_count(row: dict[str, Any]) -> int:
    return len(row.get("graphs", []) or [])


def selection_mode(row: dict[str, Any]) -> str:
    return str(row.get("ddrg_v1", {}).get("selection_mode", ""))


def anchor_source(row: dict[str, Any]) -> str:
    return str(row.get("ddrg_v1", {}).get("anchor_selection", {}).get("answer_source", ""))


def scored_value(row: dict[str, Any]) -> bool:
    return row.get("correct") is not None


def bool_mean(values: list[bool]) -> float:
    return sum(bool(value) for value in values) / len(values) if values else math.nan


def make_summary_frame(labeled_rows: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows_out = []
    labels = list(labeled_rows.keys())
    baseline_label = labels[0] if labels else ""
    paired = make_paired_examples(labeled_rows)
    pair_grouped = paired.groupby("label") if not paired.empty else {}
    for label, rows in labeled_rows.items():
        scored_rows = [row for row in rows if scored_value(row)]
        selection_counts = Counter(selection_mode(row) for row in rows if selection_mode(row))
        anchor_counts = Counter(anchor_source(row) for row in rows if anchor_source(row))
        pair_frame = pair_grouped.get_group(label) if not paired.empty and label in pair_grouped.groups else pd.DataFrame()
        rows_out.append(
            {
                "label": label,
                "baseline_label": baseline_label,
                "total_rows": len(rows),
                "scored_n": len(scored_rows),
                "error_n": sum(1 for row in rows if row.get("error")),
                "accuracy": bool_mean([bool(row.get("correct")) for row in scored_rows]),
                "oracle_rate": bool_mean([bool(row.get("oracle_hit")) for row in scored_rows if row.get("oracle_hit") is not None]),
                "avg_graphs": sum(graph_count(row) for row in rows) / len(rows) if rows else math.nan,
                "avg_probes": sum(probe_count(row) for row in rows) / len(rows) if rows else math.nan,
                "avg_unique_answers": (
                    sum(float(row.get("unique_answer_count", 0)) for row in rows) / len(rows) if rows else math.nan
                ),
                "selection_modes": json.dumps(dict(selection_counts), ensure_ascii=False, sort_keys=True),
                "anchor_source_distribution": json.dumps(dict(anchor_counts), ensure_ascii=False, sort_keys=True),
                "shared_n": int(len(pair_frame)) if not pair_frame.empty else 0,
                "wrong_to_right": int((pair_frame["transition"] == "wrong_to_right").sum()) if not pair_frame.empty else 0,
                "right_to_wrong": int((pair_frame["transition"] == "right_to_wrong").sum()) if not pair_frame.empty else 0,
                "same_answer_rate": float(pair_frame["same_answer"].mean()) if not pair_frame.empty else math.nan,
            }
        )
    return pd.DataFrame(rows_out)


def example_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("benchmark", "")), str(row.get("id", ""))


def make_paired_examples(labeled_rows: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    labels = list(labeled_rows.keys())
    if len(labels) < 2:
        return pd.DataFrame(
            columns=[
                "baseline_label",
                "label",
                "benchmark",
                "id",
                "baseline_pred",
                "pred",
                "baseline_correct",
                "correct",
                "same_answer",
                "transition",
            ]
        )
    baseline_label = labels[0]
    baseline_rows = {example_key(row): row for row in labeled_rows[baseline_label]}
    paired_rows = []
    for label in labels[1:]:
        current_rows = {example_key(row): row for row in labeled_rows[label]}
        for key in sorted(set(baseline_rows) & set(current_rows)):
            baseline = baseline_rows[key]
            current = current_rows[key]
            baseline_correct = baseline.get("correct")
            current_correct = current.get("correct")
            baseline_correct_bool = bool(baseline_correct) if baseline_correct is not None else None
            current_correct_bool = bool(current_correct) if current_correct is not None else None
            same_answer = ddrg.normalize_answer(baseline.get("pred", "")) == ddrg.normalize_answer(current.get("pred", ""))
            transition = "unknown"
            if baseline_correct_bool is True and current_correct_bool is False:
                transition = "right_to_wrong"
            elif baseline_correct_bool is False and current_correct_bool is True:
                transition = "wrong_to_right"
            elif baseline_correct_bool is not None and current_correct_bool is not None:
                transition = "same_outcome"
            paired_rows.append(
                {
                    "baseline_label": baseline_label,
                    "label": label,
                    "benchmark": key[0],
                    "id": key[1],
                    "baseline_pred": baseline.get("pred", ""),
                    "pred": current.get("pred", ""),
                    "baseline_correct": baseline_correct,
                    "correct": current_correct,
                    "same_answer": same_answer,
                    "transition": transition,
                    "baseline_selection_mode": selection_mode(baseline),
                    "selection_mode": selection_mode(current),
                    "baseline_anchor_source": anchor_source(baseline),
                    "anchor_source": anchor_source(current),
                }
            )
    return pd.DataFrame(paired_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DDRG JSONL result files or result directories.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Inputs in label=path form.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    labeled_rows = load_labeled_rows(args.inputs)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = make_summary_frame(labeled_rows)
    paired = make_paired_examples(labeled_rows)

    summary_path = output_dir / "comparison_summary.csv"
    md_path = output_dir / "comparison_summary.md"
    paired_path = output_dir / "paired_examples.csv"

    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    md_path.write_text(markdown_table(summary) + "\n", encoding="utf-8")

    print(markdown_table(summary))
    print(f"paired_examples={paired_path}")


if __name__ == "__main__":
    main()
