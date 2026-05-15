# Pilot Report 2026-05-15

Commit used for the live runs: `8a4364e` (`Add DDRG experiment runner and hybrid alignment scaffold`).

## Settings

- Model: `gpt-5.4` via Azure OpenAI
- Runner: `ddrg_v1/run_experiments.py`
- Benchmarks targeted for heuristic pilots: `logiqa`, `lsat_ar`, `mathqa`, `aiw`
- Per-benchmark limit: `10`
- Candidate graphs per example: `k=2`
- Probe budget: `max-probes=1`
- Worker count: `max-workers=1`

This is only a small pilot. `limit=10` is not enough to claim stable benchmark-level improvements.

## Completed runs

Completed pilots:

- LLM alignment + heuristic anchor: `ddrg_v1/results/pilot_llm_heuristic_limit10_20260515/`
- Hybrid alignment + heuristic anchor: `ddrg_v1/results/pilot_hybrid_heuristic_limit10_20260515/`
- LLM alignment + learned anchor: `ddrg_v1/results/pilot_llm_learned_limit10_20260515/`
  - This learned-anchor pilot was intentionally smaller and only covered `logiqa` and `lsat_ar`.

Completed sample counts:

- `pilot_llm_heuristic_limit10_20260515`: 40 scored rows total
- `pilot_hybrid_heuristic_limit10_20260515`: 40 scored rows total
- `pilot_llm_learned_limit10_20260515`: 20 scored rows total

## Summary paths

Primary summaries:

- `ddrg_v1/results/pilot_llm_heuristic_limit10_20260515/summary.csv`
- `ddrg_v1/results/pilot_llm_heuristic_limit10_20260515/summary.md`
- `ddrg_v1/results/pilot_hybrid_heuristic_limit10_20260515/summary.csv`
- `ddrg_v1/results/pilot_hybrid_heuristic_limit10_20260515/summary.md`
- `ddrg_v1/results/pilot_llm_learned_limit10_20260515/summary.csv`
- `ddrg_v1/results/pilot_llm_learned_limit10_20260515/summary.md`

Comparison outputs:

- Alignment comparison:
  - `ddrg_v1/results/comparison_alignment_limit10_20260515/comparison_summary.csv`
  - `ddrg_v1/results/comparison_alignment_limit10_20260515/comparison_summary.md`
  - `ddrg_v1/results/comparison_alignment_limit10_20260515/paired_examples.csv`
- Learned-anchor comparison:
  - `ddrg_v1/results/comparison_learned_anchor_limit10_20260515/comparison_summary.csv`
  - `ddrg_v1/results/comparison_learned_anchor_limit10_20260515/comparison_summary.md`
  - `ddrg_v1/results/comparison_learned_anchor_limit10_20260515/paired_examples.csv`

Scorer artifacts:

- Model: `ddrg_v1/results/anchor_scorer_pilot_20260515.json`
- Feature dump: `ddrg_v1/results/anchor_scorer_pilot_20260515_features.csv`

## Key observations

### 1. LLM alignment + heuristic anchor was the strongest of the two full heuristic pilots

From `pilot_llm_heuristic_limit10_20260515/summary.csv`:

- `aiw`: `0.9`
- `logiqa`: `0.5`
- `lsat_ar`: `0.9`
- `mathqa`: `0.7`
- Overall across the 40 pilot rows: `0.75`

### 2. Hybrid alignment under this conservative scaffold did not improve this pilot

From `pilot_hybrid_heuristic_limit10_20260515/summary.csv` and the alignment comparison output:

- `aiw`: `0.8`
- `logiqa`: `0.4`
- `lsat_ar`: `0.9`
- `mathqa`: `0.6`
- Overall across the 40 pilot rows: `0.675`
- In the paired comparison against the LLM-alignment baseline:
  - `wrong_to_right = 0`
  - `right_to_wrong = 3`
  - `same_answer_rate = 0.875`

This should be treated as a negative pilot result for the current hybrid scaffold, not as a final conclusion about hybrid alignment in general. The current hybrid mode is intentionally conservative and still experimental.

### 3. The preliminary learned anchor scorer is usable, but still very weakly supported

Training used 29 prior result rows and produced:

- training group anchor accuracy: `0.7391`
- held-out group anchor accuracy on the tiny `mathqa` eval slice: `0.7000`
- held-out heuristic group anchor accuracy on the same eval slice: `0.7000`

This means the scorer trained successfully, but the tiny held-out check does not show a clear advantage yet.

### 4. The small learned-anchor pilot is mildly encouraging, but too small to trust

From `pilot_llm_learned_limit10_20260515/summary.csv`:

- `logiqa`: `0.6`
- `lsat_ar`: `0.9`
- Overall across 20 pilot rows: `0.75`

Relative to the corresponding LLM-heuristic subset (`logiqa` + `lsat_ar`, 20 rows total), the learned-anchor comparison shows:

- `wrong_to_right = 1`
- `right_to_wrong = 0`
- `same_answer_rate = 0.9`

This is directionally interesting, but the scorer was trained on tiny pilot traces and should not be treated as a final selector.

## Known limitations

- All runs are small pilots with `limit=10`.
- The learned scorer is trained on a tiny and highly mixed trace pool.
- The learned-anchor pilot only covers `logiqa` and `lsat_ar`.
- The hybrid alignment mode is experimental and conservative; it is not yet an optimized hybrid adjudication pipeline.
- These runs do not justify any general claim that hybrid alignment or learned anchor selection is better overall.

## Recommended next experiment

The next reasonable step is a slightly larger but still controlled run:

1. Keep `k=2` and `max-probes=1` fixed.
2. Expand the strongest current baseline, `LLM alignment + heuristic anchor`, to `limit=25` on `logiqa`, `lsat_ar`, and `mathqa`.
3. Re-run the learned-anchor pilot on the same exact slice only after retraining the scorer on all completed pilot traces except that held-out slice.
4. Defer further hybrid claims until the alignment scaffold is tuned enough to stop losing accuracy against the plain LLM-alignment baseline.
