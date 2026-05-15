# DDRG Research Plan

Latest committed milestone inspected in this repository: `a64a430` (`Add learned anchor scorer for DDRG repair selection`).

## 1. Project Positioning

DDRG should be positioned as a verification-and-repair framework for LLM reasoning, not as SGR-style training for direct graph reasoning. The system does not train the base model to internally reason in graph form, nor does it assume that a single generated graph is a faithful representation of the model's latent reasoning. Instead, DDRG treats multiple sampled reasoning graphs as externalized, fallible traces that can be inspected and compared after generation.

In DDRG, the graph is an instrument for diagnosis and control, not the end task. The pipeline samples several candidate reasoning traces, identifies local disagreements among them, verifies selected disagreements with focused probes, repairs invalid portions of the graphs, and then uses the repaired candidates to select a final answer. The current repository state reflects this design: the learned anchor scorer added in commit `a64a430` improves only the final repaired-graph selection step and remains optional.

The right research framing is therefore: DDRG studies whether explicit multi-trace disagreement analysis can make LLM reasoning outputs more reliable under API access constraints. It is not a claim that graph-structured chain-of-thought should replace standard language generation.

## 2. Current DDRG v1 Pipeline

The current DDRG v1 implementation in `ddrg_v1/` follows a clear multi-stage pipeline.

First, the system samples multiple reasoning graphs for the same problem. These are generated independently from the base model to expose answer diversity and structural variation across candidate traces.

Second, the raw graph outputs are parsed into nodes, parent relations, edge labels, and final answers. This parsing step converts semi-structured LLM outputs into a machine-usable graph representation while retaining parse diagnostics.

Third, DDRG extracts answer-support subgraphs from each candidate. Rather than operating over every generated node equally, it focuses on the portions of each graph that feed into the final answer node.

Fourth, the system aligns local claims across candidate graphs. The intent is to identify semantically similar local steps that appear across different samples, even when wording or local structure differs.

Fifth, DDRG localizes answer-sensitive disagreements. Not every mismatch across traces matters; the pipeline tries to isolate disagreements near the causal frontier that are plausibly responsible for different final answers.

Sixth, those conflicts are verified with atomic probes. These are localized verifier prompts intended to judge narrow claims or relations rather than re-solving the entire problem.

Seventh, verified or refuted evidence is used to repair the candidate graphs. Invalid or contradicted parts are marked, and downstream consequences are propagated through the graph structure.

Eighth, DDRG selects a final anchor graph and answer from the repaired candidates. In the default mode, this still uses the existing heuristic anchor score. The new optional learned scorer can replace only this final selection mechanism.

Finally, the pipeline emits both the selected answer and a repaired graph artifact. This repaired graph is useful for trace inspection, debugging, and downstream analysis of where the method succeeded or failed.

## 3. Existing Instability Sources

The current API-based pipeline contains several known sources of instability. The first is LLM-based graph and claim alignment. Mapping locally similar claims across independently sampled traces is difficult, especially when the traces differ in decomposition, level of abstraction, or wording. Alignment inconsistency can cause downstream conflict localization to focus on the wrong evidence.

The second is probe noise. Atomic probes are narrower than full re-solving prompts, but they are still LLM judgments and can fluctuate with phrasing or prompt sensitivity. If these judgments are unreliable, repair decisions become unstable.

The third is hard repair propagation. The current repair process uses discrete verified/refuted decisions and propagates invalidity through graph structure. This can over-delete reasoning that is partially correct or only weakly connected to the refuted local claim.

The fourth is brittleness in hand-written anchor scoring. The heuristic anchor selector uses structural counts and repair-derived signals. This is reasonable as a first baseline, but it hard-codes tradeoffs between verified evidence, refuted evidence, invalid nodes, and probe support. Such weights may not generalize well across benchmarks or models.

These instability sources suggest the right order of future work. If selection improves but alignment or verification remains noisy, overall gains may be limited. The current repository should therefore be treated as a baseline for measuring where added modeling effort is useful.

## 4. Completed Milestone: Learned Anchor Scorer

The completed milestone in the current repository adds an optional learned anchor scorer for repaired-graph selection. The implementation extracts graph-level features from prior JSONL result traces. These features include structural statistics, repair summaries, probe-derived signals, answer-vote features, and alignment-support features. Training data is built offline from stored runs that include gold answers.

Training is intentionally lightweight. The current scorer is a small logistic model implemented in NumPy and trained from extracted feature vectors. This keeps the implementation inspectable and avoids introducing a larger dependency or a new end-to-end training loop.

Importantly, the heuristic anchor selector remains the default behavior. Nothing in the pipeline forces the learned scorer to be used. Inference switches to the learned selector only when `run_v1.py` is passed `--anchor-scorer-path`.

The milestone also adds diagnostics. Result JSONL traces now expose selection-time information in `anchor_selection` and `anchor_diagnostics`, making it possible to inspect why a graph was selected and what feature-based score it received.

Finally, the repository includes offline smoke coverage for this path. The smoke tests validate default heuristic selection, optional learned selection, and basic scorer training/save/load behavior. This is appropriate for the current stage, where the goal is to verify the integration rather than claim benchmark-level improvements.

## 5. Current Limitations

The present learned-scorer milestone should be described conservatively. The available training data is still tiny. The current pilot JSONL files under `ddrg_v1/results/pilot_20260514/` are enough to validate the offline path, but they are not sufficient for strong generalization claims.

The learned selector has not yet been evaluated at meaningful scale. The repository contains the machinery needed for comparison, but not enough held-out evidence to conclude that the learned scorer is consistently better than the heuristic baseline.

The scorer is graph-level only. It does not score claims, edges, or alignments directly. As a result, it can only improve the final anchor-choice step after upstream repair has already been performed.

The selector also remains dependent on upstream quality. If alignment is poor or atomic probes are noisy, the learned scorer can only react to those artifacts; it cannot correct the underlying evidence generation process.

The current features are mostly structural and repair-derived. They do not include semantic embeddings, richer textual entailment features, or model-based similarity scores between claims. This is acceptable for a minimal milestone, but it limits expressiveness.

## 6. Next Development Steps

The next stage should focus on data collection and component-wise evaluation rather than immediately adding more complexity. First, run larger pilot experiments to collect more traces across benchmarks, models, and seeds. The current selection model needs a broader corpus of repaired graphs before any serious training or ablation study is meaningful.

Second, compare heuristic versus learned anchor selection on explicitly held-out results. This comparison should be benchmark-specific and should report both answer accuracy and the frequency of each selection mode.

Third, build a hybrid alignment stage that uses deterministic normalization and candidate-pair generation first, then reserves LLM adjudication for ambiguous cases only. This should reduce variance and lower token cost while keeping flexibility where semantics are genuinely unclear.

Fourth, replace hard repair propagation with soft repair based on evidence scores. Instead of binary deletion, nodes and edges should carry confidence or support weights so that partially supported reasoning is not removed too aggressively.

Fifth, train a pairwise claim-alignment scorer. Alignment quality is an upstream bottleneck, and a lightweight pairwise scorer may be more impactful than further tuning graph-level selection.

Sixth, train or calibrate the claim/probe verifier. If probe judgments remain noisy, downstream repair and selection will inherit that noise.

Seventh, add an experiment runner that compares majority vote, DDRG with heuristic anchor selection, and DDRG with learned anchor selection under the same benchmark splits and model settings.

## 7. Suggested Immediate Experiments

Train a scorer from multiple pilot result files:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/train_anchor_scorer.py \
  ./ddrg_v1/results/pilot_20260514/logiqa.jsonl \
  ./ddrg_v1/results/pilot_20260514/lsat_ar.jsonl \
  ./ddrg_v1/results/pilot_20260514/mathqa.jsonl \
  --model-out ./ddrg_v1/results/anchor_scorer_pilot.json \
  --csv-out ./ddrg_v1/results/anchor_scorer_pilot_features.csv
```

Run held-out evaluation during training:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/train_anchor_scorer.py \
  ./ddrg_v1/results/pilot_20260514/logiqa.jsonl \
  ./ddrg_v1/results/pilot_20260514/mathqa.jsonl \
  --eval-inputs ./ddrg_v1/results/pilot_20260514/lsat_ar.jsonl \
  --model-out ./ddrg_v1/results/anchor_scorer_eval.json
```

Run DDRG inference with the optional learned scorer:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/run_v1.py \
  --benchmark logiqa \
  --limit 10 \
  --llm-provider azure \
  --model gpt-5.4 \
  --anchor-scorer-path ./ddrg_v1/results/anchor_scorer_pilot.json \
  --output ./ddrg_v1/results/logiqa_learned_anchor.jsonl
```

Extract graph-level features from prior runs:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/extract_features.py \
  ./ddrg_v1/results/pilot_20260514/logiqa.jsonl \
  ./ddrg_v1/results/pilot_20260514/lsat_ar.jsonl \
  --csv-out ./ddrg_v1/results/pilot_anchor_features.csv
```

Summarize result files for comparison:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/compare_methods.py \
  --inputs \
    heuristic=./ddrg_v1/results/pilot_heuristic/ \
    learned=./ddrg_v1/results/pilot_learned/ \
  --output-dir ./ddrg_v1/results/comparison_example
```

These experiments are sufficient for the next phase: establish whether the learned selector is worth keeping, quantify where the pipeline is still unstable, and identify whether the next research priority should be alignment, verification, repair, or final selection.
