import argparse
import math
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddrg_v1.anchor_scorer import (
    LogisticAnchorScorer,
    build_training_examples,
    feature_rows_from_examples,
    grouped_anchor_accuracy,
    grouped_heuristic_anchor_accuracy,
    load_result_rows,
)


def fmt_metric(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight learned DDRG anchor scorer from result JSONL files.")
    parser.add_argument("inputs", nargs="+", help="One or more DDRG result JSONL files with gold answers.")
    parser.add_argument("--model-out", required=True, help="Path to the output scorer JSON file.")
    parser.add_argument("--csv-out", default=None, help="Optional CSV dump of the training features.")
    parser.add_argument(
        "--eval-inputs",
        nargs="*",
        default=None,
        help="Optional JSONL files for held-out evaluation after training.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()

    paths = [Path(item).resolve() for item in args.inputs]
    rows = load_result_rows(paths)
    examples = build_training_examples(rows)
    if not examples:
        raise SystemExit("No trainable graph examples were found. Make sure the JSONL files contain gold labels.")

    scorer = LogisticAnchorScorer.fit(
        examples,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
    )

    if args.csv_out:
        frame = pd.DataFrame(feature_rows_from_examples(examples))
        csv_path = Path(args.csv_out).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)

    positives = sum(example.label for example in examples)
    learned_group_acc = grouped_anchor_accuracy(examples, scorer)
    heuristic_group_acc = grouped_heuristic_anchor_accuracy(examples)
    scorer.metadata.update(
        {
            "train_group_anchor_acc": learned_group_acc,
            "train_heuristic_group_anchor_acc": heuristic_group_acc,
        }
    )

    eval_rows = []
    eval_examples = []
    eval_heuristic_acc = math.nan
    eval_learned_acc = math.nan
    if args.eval_inputs:
        eval_paths = [Path(item).resolve() for item in args.eval_inputs]
        eval_rows = load_result_rows(eval_paths)
        eval_examples = build_training_examples(eval_rows)
        if eval_examples:
            eval_heuristic_acc = grouped_heuristic_anchor_accuracy(eval_examples)
            eval_learned_acc = grouped_anchor_accuracy(eval_examples, scorer)
            scorer.metadata.update(
                {
                    "eval_group_anchor_acc": eval_learned_acc,
                    "eval_heuristic_group_anchor_acc": eval_heuristic_acc,
                    "eval_examples": len(eval_examples),
                }
            )

    model_path = Path(args.model_out).resolve()
    scorer.save(model_path)

    print(f"rows={len(rows)}")
    print(f"graph_examples={len(examples)}")
    print(f"positive_graphs={positives}")
    print(f"heuristic_group_anchor_acc={fmt_metric(heuristic_group_acc)}")
    print(f"learned_group_anchor_acc={fmt_metric(learned_group_acc)}")
    if args.eval_inputs:
        print(f"eval_rows={len(eval_rows)}")
        print(f"eval_graph_examples={len(eval_examples)}")
        print(f"eval_heuristic_group_anchor_acc={fmt_metric(eval_heuristic_acc)}")
        print(f"eval_learned_group_anchor_acc={fmt_metric(eval_learned_acc)}")
    print(f"model_out={model_path}")
    if args.csv_out:
        print(f"feature_csv={Path(args.csv_out).resolve()}")


if __name__ == "__main__":
    main()
