# Experiment Closeout 2026-05-15

## 1. Repository State

- Repository: `haotian-io/DDRG`
- Current HEAD at closeout inspection: `ce24e29` (`Add pilot experiment report`)
- Previous report commit: `ce24e29`
- Running DDRG experiment processes found: none
- Tests:
  - `./ddrg_v1/.venv/bin/python -m py_compile ddrg_v1/*.py tests/*.py` passed
  - `./ddrg_v1/.venv/bin/python -m unittest discover -s tests -v` passed
  - `13` tests passed, `0` failed
- Code changes made in this closeout step: none
  - Only this Markdown closeout report was added.

## 2. Completed Artifacts

Existing result and comparison directories inspected from disk:

### `ddrg_v1/results/pilot_20260514/`

- Status: complete historical pilot
- Contains benchmark JSONL outputs plus `summary.csv`, `summary.md`, and `per_example.csv`
- This is the earlier small pilot already referenced in the repository documentation.

### `ddrg_v1/results/pilot_llm_heuristic_limit10_20260515/`

- Status: complete
- JSONL files:
  - `aiw.jsonl` (`10` rows)
  - `logiqa.jsonl` (`10` rows)
  - `lsat_ar.jsonl` (`10` rows)
  - `mathqa.jsonl` (`10` rows)
- Supporting files present:
  - `experiment_config.json`
  - `summary.csv`
  - `summary.md`
  - `runner_status.csv`
  - `per_example.csv`

### `ddrg_v1/results/pilot_hybrid_heuristic_limit10_20260515/`

- Status: complete
- JSONL files:
  - `aiw.jsonl` (`10` rows)
  - `logiqa.jsonl` (`10` rows)
  - `lsat_ar.jsonl` (`10` rows)
  - `mathqa.jsonl` (`10` rows)
- Supporting files present:
  - `experiment_config.json`
  - `summary.csv`
  - `summary.md`
  - `runner_status.csv`
  - `per_example.csv`

### `ddrg_v1/results/pilot_llm_learned_limit10_20260515/`

- Status: complete for the intended smaller learned-anchor pilot
- JSONL files:
  - `logiqa.jsonl` (`10` rows)
  - `lsat_ar.jsonl` (`10` rows)
- Supporting files present:
  - `experiment_config.json`
  - `summary.csv`
  - `summary.md`
  - `runner_status.csv`
  - `per_example.csv`

### `ddrg_v1/results/comparison_alignment_limit10_20260515/`

- Status: complete offline comparison artifact
- Files:
  - `comparison_summary.csv`
  - `comparison_summary.md`
  - `paired_examples.csv`

### `ddrg_v1/results/comparison_learned_anchor_limit10_20260515/`

- Status: complete offline comparison artifact
- Files:
  - `comparison_summary.csv`
  - `comparison_summary.md`
  - `paired_examples.csv`

### Local scorer artifacts

- `ddrg_v1/results/anchor_scorer_pilot_20260515.json`
- `ddrg_v1/results/anchor_scorer_pilot_20260515_features.csv`

These exist locally and were intentionally not committed.

## 3. Limit=10 Findings

The `limit=10` findings remain the same as the previous pilot report and are supported by the saved summaries.

### LLM alignment + heuristic anchor remained the strongest full heuristic pilot

From `ddrg_v1/results/pilot_llm_heuristic_limit10_20260515/summary.csv`:

- `aiw`: `0.9`
- `logiqa`: `0.5`
- `lsat_ar`: `0.9`
- `mathqa`: `0.7`

From `ddrg_v1/results/comparison_alignment_limit10_20260515/comparison_summary.csv`:

- `llm_heuristic` total rows: `40`
- `llm_heuristic` accuracy: `0.75`

### Hybrid alignment under the current conservative scaffold did not improve in this pilot

From `ddrg_v1/results/pilot_hybrid_heuristic_limit10_20260515/summary.csv`:

- `aiw`: `0.8`
- `logiqa`: `0.4`
- `lsat_ar`: `0.9`
- `mathqa`: `0.6`

From `ddrg_v1/results/comparison_alignment_limit10_20260515/comparison_summary.csv`:

- `hybrid_heuristic` total rows: `40`
- `hybrid_heuristic` accuracy: `0.675`
- `wrong_to_right = 0`
- `right_to_wrong = 3`
- `same_answer_rate = 0.875`

This is enough to keep the current hybrid alignment scaffold in the “experimental” bucket. It does not support replacing the plain LLM-alignment baseline.

### Learned anchor was mildly interesting on `logiqa + lsat_ar`, but still too small to trust

From `ddrg_v1/results/pilot_llm_learned_limit10_20260515/summary.csv`:

- `logiqa`: `0.6`
- `lsat_ar`: `0.9`

From `ddrg_v1/results/comparison_learned_anchor_limit10_20260515/comparison_summary.csv`:

- `llm_learned` total rows: `20`
- `llm_learned` accuracy: `0.75`
- `wrong_to_right = 1`
- `right_to_wrong = 0`
- `same_answer_rate = 0.9`

This remains only a small pilot signal. It is not enough evidence to claim that learned anchor selection is generally better.

## 4. Limit=25 Status

The following `limit=25` directories were checked explicitly and do not exist on disk:

- `ddrg_v1/results/pilot_llm_heuristic_limit25_20260515/`
- `ddrg_v1/results/pilot_llm_learned_lobo_limit25_logiqa_20260515/`
- `ddrg_v1/results/pilot_llm_learned_lobo_limit25_lsat_ar_20260515/`
- `ddrg_v1/results/pilot_llm_learned_lobo_limit25_mathqa_20260515/`
- `ddrg_v1/results/comparison_lobo_logiqa_limit25_20260515/`
- `ddrg_v1/results/comparison_lobo_lsat_ar_limit25_20260515/`
- `ddrg_v1/results/comparison_lobo_mathqa_limit25_20260515/`

Closeout implication:

- There are no saved `limit=25` pilot outputs to summarize.
- There are no saved `limit=25` comparison artifacts to summarize.
- This closeout therefore freezes the experiment state at the completed `limit=10` pilots plus their offline comparisons.

## 5. Current Conclusion

Conservative closeout conclusion:

- The project infrastructure is now usable for controlled pilots.
- The current evidence supports keeping `LLM alignment + heuristic anchor` as the working baseline.
- The current hybrid alignment scaffold needs tuning before more claims are made.
- The learned anchor scorer is worth further testing, but the current evidence is not strong enough to treat it as the new default.

## 6. Recommended Next Action

Recommended next step: freeze the current results and prepare a meeting update.

If a future experiment is run, it should be a single clean experiment with a fixed held-out split, rather than more ad-hoc pilot expansion.

## 7. Untracked Local Artifacts

The following local artifacts remain untracked and are intentionally not committed:

- `ddrg_v1/results/anchor_scorer_pilot_20260515.json`
- `ddrg_v1/results/anchor_scorer_pilot_20260515_features.csv`
- `ddrg_v1/results/comparison_alignment_limit10_20260515/`
- `ddrg_v1/results/comparison_learned_anchor_limit10_20260515/`
- `ddrg_v1/results/pilot_hybrid_heuristic_limit10_20260515/`
- `ddrg_v1/results/pilot_llm_heuristic_limit10_20260515/`
- `ddrg_v1/results/pilot_llm_learned_limit10_20260515/`
