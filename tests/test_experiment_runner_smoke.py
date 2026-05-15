import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ddrg_v1.run_experiments import should_skip_existing


class ExperimentRunnerSmokeTest(unittest.TestCase):
    def test_dry_run_prints_expected_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "runner"
            command = [
                sys.executable,
                "ddrg_v1/run_experiments.py",
                "--benchmarks",
                "logiqa",
                "lsat_ar",
                "--limit",
                "2",
                "--k",
                "2",
                "--max-probes",
                "1",
                "--llm-provider",
                "azure",
                "--model",
                "gpt-5.4",
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("run_v1.py --benchmark logiqa", result.stdout)
            self.assertIn("run_v1.py --benchmark lsat_ar", result.stdout)
            self.assertIn("summarize_results.py", result.stdout)

    def test_skip_existing_logic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "logiqa.jsonl"
            output_path.write_text('{"id": 1}\n', encoding="utf-8")
            self.assertTrue(should_skip_existing(output_path, skip_existing=True))
            self.assertFalse(should_skip_existing(output_path, skip_existing=False))

    def test_compare_methods_on_tiny_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            heuristic_path = root / "heuristic.jsonl"
            learned_path = root / "learned.jsonl"
            heuristic_rows = [
                {
                    "id": "1",
                    "benchmark": "logiqa",
                    "pred": "A",
                    "correct": False,
                    "oracle_hit": True,
                    "unique_answer_count": 2,
                    "graphs": [{}, {}],
                    "error": None,
                    "ddrg_v1": {
                        "probe_results": [{}],
                        "selection_mode": "verified_anchor",
                        "anchor_selection": {"answer_source": "verified_anchor"},
                    },
                }
            ]
            learned_rows = [
                {
                    "id": "1",
                    "benchmark": "logiqa",
                    "pred": "B",
                    "correct": True,
                    "oracle_hit": True,
                    "unique_answer_count": 2,
                    "graphs": [{}, {}],
                    "error": None,
                    "ddrg_v1": {
                        "probe_results": [{}],
                        "selection_mode": "learned_anchor",
                        "anchor_selection": {"answer_source": "learned_anchor"},
                    },
                }
            ]
            heuristic_path.write_text("\n".join(json.dumps(row) for row in heuristic_rows) + "\n", encoding="utf-8")
            learned_path.write_text("\n".join(json.dumps(row) for row in learned_rows) + "\n", encoding="utf-8")

            output_dir = root / "comparison"
            command = [
                sys.executable,
                "ddrg_v1/compare_methods.py",
                "--inputs",
                f"heuristic={heuristic_path}",
                f"learned={learned_path}",
                "--output-dir",
                str(output_dir),
            ]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((output_dir / "comparison_summary.csv").exists())
            self.assertTrue((output_dir / "paired_examples.csv").exists())
            with (output_dir / "comparison_summary.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            learned_row = next(row for row in rows if row["label"] == "learned")
            self.assertEqual(learned_row["wrong_to_right"], "1")

    def test_compare_methods_accepts_directory_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            heuristic_dir = root / "heuristic"
            learned_dir = root / "learned"
            heuristic_dir.mkdir()
            learned_dir.mkdir()
            (heuristic_dir / "part1.jsonl").write_text(
                json.dumps(
                    {
                        "id": "1",
                        "benchmark": "logiqa",
                        "pred": "A",
                        "correct": False,
                        "graphs": [{}],
                        "ddrg_v1": {"probe_results": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (learned_dir / "part1.jsonl").write_text(
                json.dumps(
                    {
                        "id": "1",
                        "benchmark": "logiqa",
                        "pred": "A",
                        "correct": True,
                        "graphs": [{}, {}],
                        "ddrg_v1": {"probe_results": [{}], "selection_mode": "learned_anchor"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "comparison_dir"
            command = [
                sys.executable,
                "ddrg_v1/compare_methods.py",
                "--inputs",
                f"heuristic={heuristic_dir}",
                f"learned={learned_dir}",
                "--output-dir",
                str(output_dir),
            ]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((output_dir / "comparison_summary.md").exists())
            self.assertIn("same_answer_rate", (output_dir / "comparison_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
