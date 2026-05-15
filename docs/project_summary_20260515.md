# DDRG Project Summary

Last inspected repository commit: `94651fd`

## 1. Project Positioning

This repository is the working tree for **DDRG: Disagreement-Driven Reasoning Graph Repair**.

The goal of DDRG is **not** to train a model to directly solve problems by graph reasoning. Instead, the project treats reasoning graphs as an external diagnostic and correction layer around an LLM pipeline:

1. sample multiple reasoning graphs,
2. parse them into structured graph form,
3. align semantically similar local claims,
4. localize answer-sensitive disagreements,
5. verify narrow conflicts with probes,
6. repair invalid graph parts, and
7. select the most trustworthy repaired graph / final answer.

The current codebase already supports:

- default LLM-based alignment,
- experimental conservative hybrid alignment,
- default heuristic anchor selection,
- optional learned anchor selection through a lightweight graph-level scorer,
- offline feature extraction, scorer training, experiment running, and method comparison.

## 2. Repository Layout

Current top-level structure:

```text
DDRG/
├── benchmarks/
├── ddrg_v1/
├── docs/
└── tests/
```

### `benchmarks/`

Prepared local benchmark files expected by `ddrg_v1/run_v1.py`:

- `benchmarks/aiw/AIW_easy.pkl`
- `benchmarks/aiw+/AIW_hard.pkl`
- `benchmarks/logiqa/logiqa_test.csv`
- `benchmarks/lsat_ar/lsat-ar.csv`
- `benchmarks/mathqa/mathqa.csv`
- `benchmarks/medqa/medqa.csv`

### `ddrg_v1/`

Main implementation package and scripts.

Key files:

- `core.py`: core DDRG pipeline orchestration
- `run_v1.py`: single benchmark / single question entrypoint
- `llm.py`: OpenAI-compatible and Azure client adapter
- `alignment.py`: conservative hybrid alignment scaffold
- `anchor_scorer.py`: graph-level feature extraction and lightweight learned selector
- `train_anchor_scorer.py`: offline scorer training
- `extract_features.py`: feature dump from saved traces
- `run_experiments.py`: controlled multi-benchmark pilot runner
- `compare_methods.py`: offline comparison of result directories / JSONL outputs
- `summarize_results.py`: summary table generation from JSONL outputs
- `download_benchmarks.py`: benchmark preparation helper
- `prompts.py`: prompt templates
- `utils.py`: parsing, dataset loading, normalization, and graph utilities

### `ddrg_v1/results/`

Local experiment outputs and smoke artifacts.

Tracked historical outputs:

- `pilot_20260514/`
- `aiw_smoke.jsonl`
- `logiqa_smoke.jsonl`
- `smoke_adhoc.jsonl`

Local generated pilot outputs currently present:

- `pilot_llm_heuristic_limit10_20260515/`
- `pilot_hybrid_heuristic_limit10_20260515/`
- `pilot_llm_learned_limit10_20260515/`
- `comparison_alignment_limit10_20260515/`
- `comparison_learned_anchor_limit10_20260515/`
- `anchor_scorer_pilot_20260515.json`
- `anchor_scorer_pilot_20260515_features.csv`

These locally generated pilot artifacts are now ignored in `.gitignore` so the repository can stay clean without deleting them.

### `docs/`

Current documentation set:

- `research_plan.md`: project framing, current bottlenecks, and next-step plan
- `pilot_report_20260515.md`: focused report for the completed limit=10 pilot
- `experiment_closeout_20260515.md`: closeout note freezing the pilot state
- `project_summary_20260515.md`: this broader repository-wide summary

### `tests/`

Current smoke coverage:

- `test_anchor_scorer_smoke.py`
- `test_experiment_runner_smoke.py`
- `test_hybrid_alignment_smoke.py`

## 3. Code Workflow

### Main online workflow

For live DDRG inference:

- `ddrg_v1/run_v1.py`

This supports:

- benchmark mode via `--benchmark`
- single-question mode via `--question`
- default `--alignment-mode llm`
- optional `--alignment-mode hybrid`
- optional `--anchor-scorer-path <model.json>`

### Controlled pilot workflow

For multi-benchmark pilot execution:

- `ddrg_v1/run_experiments.py`

This writes:

- one JSONL file per benchmark,
- `experiment_config.json`,
- `summary.csv`,
- `summary.md`,
- `per_example.csv`,
- `runner_status.csv`

### Offline analysis workflow

For offline comparison and model training:

- `ddrg_v1/compare_methods.py`
- `ddrg_v1/train_anchor_scorer.py`
- `ddrg_v1/extract_features.py`
- `ddrg_v1/summarize_results.py`

These tools do not require API keys if they are run only on saved outputs.

## 4. Current Experiment State

### Historical pilot

`ddrg_v1/results/pilot_20260514/` is the earlier small cross-benchmark pilot.

It covers:

- `aiw`
- `aiw+`
- `logiqa`
- `lsat_ar`
- `mathqa`
- `medqa`

This pilot is useful mainly as seed data and as the first saved end-to-end trace set.

### Current limit=10 controlled pilots

#### `pilot_llm_heuristic_limit10_20260515/`

Complete on:

- `aiw` (`10`)
- `logiqa` (`10`)
- `lsat_ar` (`10`)
- `mathqa` (`10`)

Summary from `summary.csv`:

- `aiw`: `0.9`
- `logiqa`: `0.5`
- `lsat_ar`: `0.9`
- `mathqa`: `0.7`

#### `pilot_hybrid_heuristic_limit10_20260515/`

Complete on:

- `aiw` (`10`)
- `logiqa` (`10`)
- `lsat_ar` (`10`)
- `mathqa` (`10`)

Summary from `summary.csv`:

- `aiw`: `0.8`
- `logiqa`: `0.4`
- `lsat_ar`: `0.9`
- `mathqa`: `0.6`

#### `pilot_llm_learned_limit10_20260515/`

Complete on the intentionally smaller learned-anchor slice:

- `logiqa` (`10`)
- `lsat_ar` (`10`)

Summary from `summary.csv`:

- `logiqa`: `0.6`
- `lsat_ar`: `0.9`

### Offline comparisons

#### `comparison_alignment_limit10_20260515/`

Compares:

- `llm_heuristic`
- `hybrid_heuristic`

Key summary values:

- LLM heuristic accuracy: `0.75`
- Hybrid heuristic accuracy: `0.675`
- `wrong_to_right = 0`
- `right_to_wrong = 3`
- `same_answer_rate = 0.875`

Interpretation:

- The current conservative hybrid scaffold did **not** improve over the plain LLM-alignment baseline in this pilot.

#### `comparison_learned_anchor_limit10_20260515/`

Compares:

- `llm_heuristic`
- `llm_learned`

Key summary values on the shared `logiqa + lsat_ar` slice:

- LLM learned accuracy: `0.75`
- `wrong_to_right = 1`
- `right_to_wrong = 0`
- `same_answer_rate = 0.9`

Interpretation:

- The learned anchor selector is mildly interesting, but the slice is too small to justify stronger claims.

### Scorer artifact

Local preliminary learned-anchor scorer:

- `ddrg_v1/results/anchor_scorer_pilot_20260515.json`

Associated feature dump:

- `ddrg_v1/results/anchor_scorer_pilot_20260515_features.csv`

This scorer should be treated as a pilot artifact, not a final model.

## 5. Current Technical Conclusion

The current repository supports controlled DDRG experiments end-to-end, including:

- live inference,
- trace logging,
- hybrid alignment diagnostics,
- graph-level feature extraction,
- lightweight learned anchor scoring,
- runner-based multi-benchmark pilots,
- offline method comparison,
- smoke-test validation.

The strongest supported baseline at the moment remains:

- **LLM alignment + heuristic anchor selection**

The current evidence supports:

- keeping the heuristic baseline as the default,
- treating hybrid alignment as still experimental,
- continuing to evaluate learned anchor selection only in controlled held-out settings.

## 6. Current Validation Status

Local validation at inspection time:

- `python -m py_compile ddrg_v1/*.py tests/*.py`: passed
- `python -m unittest discover -s tests -v`: passed
- test count: `13`

This means the repo is in a runnable and internally consistent state for local development.

## 7. File Organization Notes

Current organization is already workable. The most important cleanup completed here is:

- generated pilot outputs are ignored locally instead of repeatedly showing up as untracked noise.

Suggested conventions going forward:

1. Keep repository-tracked docs in `docs/`.
2. Keep reusable benchmark assets in `benchmarks/`.
3. Keep executable method code and offline tooling under `ddrg_v1/`.
4. Keep large experiment outputs local unless there is a clear reason to version them.
5. Use one Markdown report per meaningful experiment milestone rather than embedding changing status into multiple README files.

## 8. Recommended Next Step

If the goal is stability rather than more ad-hoc exploration, the next step should be:

1. freeze the current `limit=10` pilot state,
2. prepare a short meeting/update deck from the saved reports,
3. design one clean future experiment with a fixed held-out split for learned-anchor evaluation,
4. avoid expanding hybrid alignment claims until the scaffold is tuned enough to beat the plain LLM-alignment baseline on a controlled slice.
