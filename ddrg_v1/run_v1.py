import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from ddrg_v1 import utils as ddrg
from ddrg_v1.anchor_scorer import load_anchor_scorer
from ddrg_v1.core import run_ddrg_v1
from ddrg_v1.llm import make_llm_client

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_model(args: argparse.Namespace) -> str:
    provider = args.llm_provider.strip().lower()
    if provider in {"azure", "azure-openai", "azure_openai"}:
        if args.azure_deployment and args.model == DEFAULT_MODEL:
            return args.azure_deployment
    return args.model


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.question:
        return [
            {
                "id": "adhoc-1",
                "question": args.question,
                "gold": ddrg.normalize_answer(args.gold) if args.gold is not None else None,
                "raw_gold": args.gold,
            }
        ]
    return ddrg.load_dataset(args.benchmark, args.limit, args.seed)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    args.model = resolve_model(args)
    args.anchor_scorer = (
        load_anchor_scorer(args.anchor_scorer_path) if getattr(args, "anchor_scorer_path", None) else None
    )
    client = make_llm_client(
        provider=args.llm_provider,
        base_url=args.base_url,
        api_key=args.api_key,
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version,
        azure_deployment=args.azure_deployment,
        referer=args.referer,
        title=args.title,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    if (
        args.llm_provider.strip().lower() in {"azure", "azure-openai", "azure_openai"}
        and args.model == DEFAULT_MODEL
        and getattr(client, "azure_deployment", None)
    ):
        args.model = client.azure_deployment
    rows = load_rows(args)
    run_name = args.benchmark or "adhoc"
    output_path = Path(args.output)
    correct = 0
    scored_rows = 0

    for idx, row in enumerate(tqdm(rows, desc=f"{run_name}:ddrg-v1"), start=1):
        error_message = None
        try:
            pred, outputs, candidates, info = run_ddrg_v1(client, args, row["question"])
            sample_answers = [candidate.parsed.final_answer for candidate in candidates]
            gold = row["gold"]
            is_correct = ddrg.answers_match(pred, gold) if gold is not None else None
            oracle_hit = (
                any(ddrg.answers_match(answer, gold) for answer in sample_answers) if gold is not None else None
            )
            if is_correct is not None:
                correct += int(is_correct)
                scored_rows += 1
        except Exception as exc:
            pred = ""
            outputs = []
            candidates = []
            info = {"error": str(exc)}
            sample_answers = []
            gold = row["gold"]
            is_correct = None
            oracle_hit = None
            error_message = str(exc)

        result = {
            "id": row["id"],
            "benchmark": args.benchmark or "adhoc",
            "method": "ddrg_v1_clean",
            "model": args.model,
            "question": row["question"],
            "gold": gold,
            "raw_gold": str(row["raw_gold"]),
            "pred": pred,
            "correct": is_correct,
            "sample_answers": sample_answers,
            "unique_answer_count": len(set(answer for answer in sample_answers if answer)),
            "oracle_hit": oracle_hit,
            "outputs": outputs,
            "graphs": [
                {
                    "final_answer": candidate.parsed.final_answer,
                    "raw_final_answer": candidate.parsed.raw_final_answer,
                    "parse_ok": candidate.parsed.parse_ok,
                    "issues": candidate.parsed.issues,
                    "score": candidate.score,
                }
                for candidate in candidates
            ],
            "ddrg_v1": info,
            "error": error_message,
        }
        append_jsonl(output_path, result)

        if args.print_each:
            if error_message is not None:
                print(
                    f"[{idx}/{len(rows)}] id={row['id']} ERROR {error_message}",
                    flush=True,
                )
            elif is_correct is None:
                print(
                    f"[{idx}/{len(rows)}] id={row['id']} pred={pred!r} graphs={len(candidates)}",
                    flush=True,
                )
            else:
                print(
                    f"[{idx}/{len(rows)}] id={row['id']} "
                    f"{'OK' if is_correct else 'NO'} pred={pred!r} gold={gold!r} "
                    f"acc={correct / idx:.4f} oracle={'Y' if oracle_hit else 'N'} "
                    f"graphs={len(candidates)}",
                    flush=True,
                )
            if args.print_samples:
                print(f"  samples={sample_answers}", flush=True)
            if args.print_info:
                print(f"  ddrg_v1={info}", flush=True)
            if args.print_raw_graphs:
                for graph_idx, output in enumerate(outputs, start=1):
                    print(f"\n--- Raw sampled graph {graph_idx} ---\n{output}", flush=True)
            if args.print_graph:
                graph_text = info.get("integrated_repaired_graph_text", "")
                print(f"\n--- Repaired reasoning graph F' ---\n{graph_text or '<empty>'}", flush=True)

    if scored_rows:
        print(f"Done. Accuracy: {correct}/{scored_rows} = {correct / scored_rows:.4f}")
    else:
        print(f"Done. Processed {len(rows)} row(s) without gold labels.")
    print(f"Results saved to: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean DDRG v1 main-method runner.")
    parser.add_argument("--benchmark", choices=sorted(ddrg.BENCHMARK_FILES))
    parser.add_argument("--question", default=None, help="Run a single ad-hoc question instead of a benchmark.")
    parser.add_argument("--gold", default=None, help="Optional gold answer for --question mode.")
    parser.add_argument("--output", default="results/ddrg_v1.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--llm-provider", default="openai-compatible")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-version", default=None)
    parser.add_argument("--azure-deployment", default=None)
    parser.add_argument("--referer", default="https://localhost")
    parser.add_argument("--title", default="ddrg-v1")
    parser.add_argument(
        "--anchor-scorer-path",
        default=None,
        help="Optional path to a trained learned anchor scorer JSON file.",
    )

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-graph-nodes", type=int, default=20)
    parser.add_argument("--max-support-nodes", type=int, default=20)
    parser.add_argument("--max-graph-chars", type=int, default=2200)

    parser.add_argument("--max-probes", type=int, default=2)
    parser.add_argument("--probe-votes", type=int, default=1)
    parser.add_argument("--builder-max-tokens", type=int, default=1200)
    parser.add_argument("--probe-max-tokens", type=int, default=256)
    parser.add_argument("--integration-max-chars", type=int, default=7000)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--probe-temperature", type=float, default=0.0)

    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--print-each", action="store_true")
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument("--print-info", action="store_true")
    parser.add_argument("--print-graph", action="store_true")
    parser.add_argument("--print-raw-graphs", action="store_true")
    return parser


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    if bool(parsed_args.benchmark) == bool(parsed_args.question):
        raise SystemExit("Pass exactly one of --benchmark or --question.")
    run(parsed_args)
