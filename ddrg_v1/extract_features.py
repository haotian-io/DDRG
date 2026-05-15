import argparse
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddrg_v1.anchor_scorer import build_training_examples, feature_rows_from_examples, load_result_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-graph DDRG anchor scorer features from result JSONL files.")
    parser.add_argument("inputs", nargs="+", help="One or more DDRG result JSONL files.")
    parser.add_argument("--csv-out", required=True, help="Output CSV path for extracted graph features.")
    args = parser.parse_args()

    paths = [Path(item).resolve() for item in args.inputs]
    rows = load_result_rows(paths)
    examples = build_training_examples(rows)
    frame = pd.DataFrame(feature_rows_from_examples(examples))

    output_path = Path(args.csv_out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    print(f"rows={len(rows)}")
    print(f"graph_examples={len(examples)}")
    print(f"feature_csv={output_path}")


if __name__ == "__main__":
    main()
