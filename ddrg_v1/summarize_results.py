import argparse
import json
from pathlib import Path

import pandas as pd


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def selection_mode(row: dict) -> str:
    return str(row.get("ddrg_v1", {}).get("selection_mode", ""))


def probe_count(row: dict) -> int:
    return len(row.get("ddrg_v1", {}).get("probe_results", []))


def make_summary_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "benchmark",
                "model",
                "method",
                "n",
                "scored_n",
                "error_n",
                "accuracy",
                "oracle_rate",
                "avg_unique_answers",
                "avg_graphs",
                "avg_probes",
                "selection_modes",
                "source_files",
            ]
        )
    df["scored"] = df["correct"].notna()
    df["errored"] = df["error"].notna() if "error" in df.columns else False
    df["correct_num"] = df["correct"].fillna(False).astype(int)
    df["oracle_num"] = df["oracle_hit"].fillna(False).astype(int)
    df["graph_count"] = df["graphs"].apply(len)
    df["probe_count"] = df.apply(probe_count, axis=1)
    df["selection_mode"] = df.apply(selection_mode, axis=1)

    grouped = []
    for (benchmark, model, method), group in df.groupby(["benchmark", "model", "method"], dropna=False):
        mode_counts = group["selection_mode"].value_counts().to_dict()
        grouped.append(
            {
                "benchmark": benchmark,
                "model": model,
                "method": method,
                "n": int(len(group)),
                "scored_n": int(group["scored"].sum()),
                "error_n": int(group["errored"].sum()),
                "accuracy": group.loc[group["scored"], "correct_num"].mean() if group["scored"].any() else float("nan"),
                "oracle_rate": group.loc[group["scored"], "oracle_num"].mean() if group["scored"].any() else float("nan"),
                "avg_unique_answers": group["unique_answer_count"].mean(),
                "avg_graphs": group["graph_count"].mean(),
                "avg_probes": group["probe_count"].mean(),
                "selection_modes": json.dumps(mode_counts, ensure_ascii=False, sort_keys=True),
                "source_files": ", ".join(sorted(set(group["_source_file"]))),
            }
        )
    summary = pd.DataFrame(grouped).sort_values(["benchmark", "model", "method"]).reset_index(drop=True)
    return summary


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in ["accuracy", "oracle_rate", "avg_unique_answers", "avg_graphs", "avg_probes"]:
        display[col] = display[col].map(lambda x: "nan" if pd.isna(x) else f"{x:.3f}")
    columns = list(display.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in display.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DDRG v1 experiment JSONL files.")
    parser.add_argument("inputs", nargs="+", help="One or more JSONL result files.")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output path.")
    parser.add_argument("--md-out", default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    paths = [Path(item).resolve() for item in args.inputs]
    rows = load_rows(paths)
    summary = make_summary_frame(rows)

    if args.csv_out:
        csv_path = Path(args.csv_out).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(csv_path, index=False)
    if args.md_out:
        md_path = Path(args.md_out).resolve()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown_table(summary) + "\n", encoding="utf-8")

    print(markdown_table(summary))


if __name__ == "__main__":
    main()
