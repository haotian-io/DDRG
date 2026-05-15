# DDRG v1

## Pipeline

1. Sample K independent reasoning graphs.
2. Parse graph text into nodes and edges.
3. Extract each graph's ANS-support subgraph.
4. Align similar local steps across all support graphs.
5. Find answer-sensitive local disagreements.
6. Verify each disagreement with an atomic yes/no probe.
7. Mark verified/refuted claims or edges.
8. Propagate invalid downstream nodes.
9. Select the best valid repaired graph as the verified anchor.
10. Output the anchor answer and a programmatic repaired graph `F'`.

## Run

From the repository root:

```bash
python -m venv ./ddrg_v1/.venv
./ddrg_v1/.venv/bin/python -m pip install -r ./ddrg_v1/requirements.txt
```

Prepare benchmark files under `./benchmarks`:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/download_benchmarks.py
```

This writes the files expected by `run_v1.py`:

- `./benchmarks/logiqa/logiqa_test.csv`
- `./benchmarks/lsat_ar/lsat-ar.csv`
- `./benchmarks/mathqa/mathqa.csv`
- `./benchmarks/medqa/medqa.csv`
- `./benchmarks/aiw/AIW_easy.pkl`
- `./benchmarks/aiw+/AIW_hard.pkl`

OpenRouter / OpenAI-compatible:

```bash
export OPENROUTER_API_KEY="..."
./ddrg_v1/.venv/bin/python ./ddrg_v1/run_v1.py \
  --benchmark logiqa \
  --limit 10 \
  --k 3 \
  --max-probes 2 \
  --print-each \
  --print-graph \
  --output results/ddrg_v1_test.jsonl
```

Azure OpenAI:

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2025-01-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-5.4"

./ddrg_v1/.venv/bin/python ./ddrg_v1/run_v1.py \
  --llm-provider azure \
  --model gpt-5.4 \
  --benchmark logiqa \
  --limit 10
```

Single-question smoke test without benchmark files:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/run_v1.py \
  --question "What is 1 + 1? Answer with the number only." \
  --llm-provider azure \
  --model gpt-5.4 \
  --k 1 \
  --max-workers 1 \
  --max-probes 0 \
  --print-each \
  --print-graph \
  --output ./ddrg_v1/results/smoke_adhoc.jsonl
```

## LLM Adapter

All model calls go through [llm.py](./llm.py). The method code only expects an
object with:

```python
client.generate(content, model, temperature, top_p, max_tokens) -> str
```

To use another platform or a local model server, add a new adapter in
`llm.py` and return it from `make_llm_client`.

Notes:

- `run_v1.py` now supports exactly one of `--benchmark` or `--question`.
- If Azure environment variables are stored in a repository-adjacent `.env`,
  `llm.py` will auto-load them.
- If a benchmark file is missing, the loader raises a clear error and suggests
  `--question` mode for smoke testing.

Core knobs:

- `--k`: number of sampled graphs.
- `--max-probes`: maximum localized disagreements to verify.
- `--probe-votes`: verifier calls per probe.
- `--max-support-nodes`: maximum ANS-support nodes per graph.
- `--max-graph-chars`: maximum prompt characters per support graph.
- `--print-graph`: print the repaired `F'` graph.
- `--print-raw-graphs`: print raw sampled graphs.

## Learned Anchor Scorer

You can train a lightweight learned anchor scorer from prior JSONL runs that
contain gold answers:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/train_anchor_scorer.py \
  ./ddrg_v1/results/pilot_20260514/logiqa.jsonl \
  ./ddrg_v1/results/pilot_20260514/lsat_ar.jsonl \
  --model-out ./ddrg_v1/results/anchor_scorer_logreg.json \
  --csv-out ./ddrg_v1/results/anchor_scorer_features.csv
```

To report held-out anchor-selection accuracy on separate result files:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/train_anchor_scorer.py \
  ./ddrg_v1/results/pilot_20260514/logiqa.jsonl \
  --eval-inputs ./ddrg_v1/results/pilot_20260514/lsat_ar.jsonl \
  --model-out ./ddrg_v1/results/anchor_scorer_logreg.json
```

To inspect the per-graph features without training:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/extract_features.py \
  ./ddrg_v1/results/pilot_20260514/logiqa.jsonl \
  --csv-out ./ddrg_v1/results/logiqa_anchor_features.csv
```

To use a trained scorer at inference time, pass it to `run_v1.py`:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/run_v1.py \
  --benchmark logiqa \
  --limit 10 \
  --llm-provider azure \
  --model gpt-5.4 \
  --anchor-scorer-path ./ddrg_v1/results/anchor_scorer_logreg.json
```

When `--anchor-scorer-path` is set, the result JSONL includes learned-anchor
diagnostics under `ddrg_v1.anchor_selection`, `ddrg_v1.anchor_diagnostics`, and
the final `trace` stage. Without that flag, the default heuristic anchor path
is preserved.
