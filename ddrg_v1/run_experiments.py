import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddrg_v1 import utils as ddrg
from ddrg_v1.summarize_results import load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled DDRG pilot experiments across benchmarks.")
    parser.add_argument("--benchmarks", nargs="*", choices=sorted(ddrg.BENCHMARK_FILES), default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-probes", type=int, default=2)
    parser.add_argument("--llm-provider", default="openai-compatible")
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--anchor-scorer-path", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--extra-run-v1-args", default="")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def benchmark_output_path(output_dir: Path, benchmark: str) -> Path:
    return output_dir / f"{benchmark}.jsonl"


def should_skip_existing(output_path: Path, skip_existing: bool) -> bool:
    return bool(skip_existing and output_path.exists() and output_path.stat().st_size > 0)


def build_run_command(args: argparse.Namespace, benchmark: str, output_path: Path) -> list[str]:
    command = [
        args.python_bin,
        str(Path(__file__).resolve().with_name("run_v1.py")),
        "--benchmark",
        benchmark,
        "--output",
        str(output_path),
        "--llm-provider",
        args.llm_provider,
        "--model",
        args.model,
        "--k",
        str(args.k),
        "--max-probes",
        str(args.max_probes),
        "--max-workers",
        str(args.max_workers),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.anchor_scorer_path:
        command.extend(["--anchor-scorer-path", args.anchor_scorer_path])
    if args.extra_run_v1_args:
        command.extend(shlex.split(args.extra_run_v1_args))
    return command


def build_summarize_command(output_dir: Path, jsonl_paths: list[Path], python_bin: str) -> list[str]:
    return [
        python_bin,
        str(Path(__file__).resolve().with_name("summarize_results.py")),
        *[str(path) for path in jsonl_paths],
        "--csv-out",
        str(output_dir / "summary.csv"),
        "--md-out",
        str(output_dir / "summary.md"),
    ]


def save_experiment_config(args: argparse.Namespace, output_dir: Path) -> Path:
    config = {
        "experiment_name": args.experiment_name or output_dir.name,
        "benchmarks": args.benchmarks,
        "limit": args.limit,
        "k": args.k,
        "max_probes": args.max_probes,
        "llm_provider": args.llm_provider,
        "model": args.model,
        "max_workers": args.max_workers,
        "anchor_scorer_path": args.anchor_scorer_path,
        "resume": args.resume,
        "skip_existing": args.skip_existing,
        "summarize_only": args.summarize_only,
        "extra_run_v1_args": args.extra_run_v1_args,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path


def render_command(command: list[str]) -> str:
    return shlex.join(command)


def format_status_table(rows: list[dict[str, Any]]) -> str:
    headers = ["benchmark", "status", "output", "detail"]
    body = [
        [
            str(row.get("benchmark", "")),
            str(row.get("status", "")),
            str(row.get("output", "")),
            str(row.get("detail", "")),
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(line[idx]) for line in body)) if body else len(header)
        for idx, header in enumerate(headers)
    ]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in body:
        lines.append(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return "\n".join(lines)


def collect_jsonl_paths(output_dir: Path, benchmarks: list[str]) -> list[Path]:
    if benchmarks:
        paths = [benchmark_output_path(output_dir, benchmark) for benchmark in benchmarks]
        return [path for path in paths if path.exists() and path.stat().st_size > 0]
    return sorted(path for path in output_dir.glob("*.jsonl") if path.stat().st_size > 0)


def write_runner_status(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    status_path = output_dir / "runner_status.csv"
    with status_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark", "status", "output", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    return status_path


def write_per_example_csv(output_dir: Path, jsonl_paths: list[Path]) -> Path:
    rows = load_rows(jsonl_paths)
    per_example_path = output_dir / "per_example.csv"
    fieldnames = [
        "benchmark",
        "model",
        "method",
        "id",
        "pred",
        "gold",
        "correct",
        "oracle_hit",
        "unique_answer_count",
        "graph_count",
        "probe_count",
        "selection_mode",
        "anchor_source",
        "error",
        "source_file",
    ]
    with per_example_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "benchmark": row.get("benchmark", ""),
                    "model": row.get("model", ""),
                    "method": row.get("method", ""),
                    "id": row.get("id", ""),
                    "pred": row.get("pred", ""),
                    "gold": row.get("gold", ""),
                    "correct": row.get("correct"),
                    "oracle_hit": row.get("oracle_hit"),
                    "unique_answer_count": row.get("unique_answer_count"),
                    "graph_count": len(row.get("graphs", []) or []),
                    "probe_count": len(row.get("ddrg_v1", {}).get("probe_results", [])),
                    "selection_mode": row.get("ddrg_v1", {}).get("selection_mode", ""),
                    "anchor_source": row.get("ddrg_v1", {}).get("anchor_selection", {}).get("answer_source", ""),
                    "error": row.get("error"),
                    "source_file": row.get("_source_file", ""),
                }
            )
    return per_example_path


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def main() -> None:
    args = parse_args()
    skip_existing = bool(args.skip_existing or args.resume)
    output_dir = Path(args.output_dir).resolve()
    if not args.summarize_only and not args.benchmarks:
        raise SystemExit("Pass at least one benchmark unless --summarize-only is set.")

    status_rows: list[dict[str, Any]] = []
    if args.dry_run:
        print(f"# experiment_name={args.experiment_name or output_dir.name}")

    if not args.dry_run:
        config_path = save_experiment_config(args, output_dir)
        print(f"Saved config: {config_path}")

    planned_output_paths: list[Path] = []
    if not args.summarize_only:
        for benchmark in args.benchmarks:
            output_path = benchmark_output_path(output_dir, benchmark)
            planned_output_paths.append(output_path)
            if should_skip_existing(output_path, skip_existing):
                status_rows.append(
                    {
                        "benchmark": benchmark,
                        "status": "skipped_existing",
                        "output": str(output_path),
                        "detail": "existing non-empty output",
                    }
                )
                continue

            command = build_run_command(args, benchmark, output_path)
            if args.dry_run:
                print(render_command(command))
                status_rows.append(
                    {
                        "benchmark": benchmark,
                        "status": "dry_run",
                        "output": str(output_path),
                        "detail": "command printed only",
                    }
                )
                continue

            result = run_subprocess(command)
            if result.returncode == 0:
                status_rows.append(
                    {
                        "benchmark": benchmark,
                        "status": "ok",
                        "output": str(output_path),
                        "detail": (result.stdout.strip().splitlines() or ["completed"])[-1],
                    }
                )
                continue

            detail = result.stderr.strip() or result.stdout.strip() or f"exit_code={result.returncode}"
            status_rows.append(
                {
                    "benchmark": benchmark,
                    "status": "failed",
                    "output": str(output_path),
                    "detail": detail.splitlines()[-1],
                }
            )
            if args.fail_fast:
                break

    jsonl_paths = planned_output_paths if args.dry_run else collect_jsonl_paths(output_dir, args.benchmarks)
    if jsonl_paths:
        summarize_command = build_summarize_command(output_dir, jsonl_paths, args.python_bin)
        if args.dry_run:
            print(render_command(summarize_command))
            status_rows.append(
                {
                    "benchmark": "<summary>",
                    "status": "dry_run",
                    "output": str(output_dir / "summary.csv"),
                    "detail": "summary command printed only",
                }
            )
        else:
            result = run_subprocess(summarize_command)
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or f"exit_code={result.returncode}"
                status_rows.append(
                    {
                        "benchmark": "<summary>",
                        "status": "failed",
                        "output": str(output_dir / "summary.csv"),
                        "detail": detail.splitlines()[-1],
                    }
                )
                if args.fail_fast:
                    print(format_status_table(status_rows))
                    write_runner_status(output_dir, status_rows)
                    raise SystemExit(result.returncode)
            else:
                per_example_path = write_per_example_csv(output_dir, jsonl_paths)
                status_rows.append(
                    {
                        "benchmark": "<summary>",
                        "status": "ok" if not args.dry_run else "dry_run",
                        "output": str(output_dir / "summary.csv"),
                        "detail": f"summary files written; per-example at {per_example_path.name}",
                    }
                )
    elif not args.dry_run:
        status_rows.append(
            {
                "benchmark": "<summary>",
                "status": "skipped",
                "output": str(output_dir / "summary.csv"),
                "detail": "no JSONL outputs found",
            }
        )

    print(format_status_table(status_rows))
    if not args.dry_run:
        status_path = write_runner_status(output_dir, status_rows)
        print(f"Runner status: {status_path}")


if __name__ == "__main__":
    main()
