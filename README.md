# DDRG

Merged working tree for the current DDRG v1 project.

This repository now contains:

- `ddrg_v1/`: the cleaned DDRG v1 pipeline, Azure/OpenAI-compatible LLM adapter, benchmark downloader, runner, and result summarizer.
- `benchmarks/`: local benchmark files prepared for `aiw`, `aiw+`, `logiqa`, `lsat_ar`, `mathqa`, and `medqa`.
- `ddrg_v1/results/`: smoke-test outputs and a cross-benchmark pilot run.

## Layout

```text
DDRG/
├── benchmarks/
│   ├── aiw/AIW_easy.pkl
│   ├── aiw+/AIW_hard.pkl
│   ├── logiqa/logiqa_test.csv
│   ├── lsat_ar/lsat-ar.csv
│   ├── mathqa/mathqa.csv
│   └── medqa/medqa.csv
└── ddrg_v1/
    ├── core.py
    ├── llm.py
    ├── run_v1.py
    ├── download_benchmarks.py
    ├── summarize_results.py
    └── results/
```

## Quick Start

From the repository root:

```bash
python -m venv ./ddrg_v1/.venv
./ddrg_v1/.venv/bin/python -m pip install -r ./ddrg_v1/requirements.txt
./ddrg_v1/.venv/bin/python ./ddrg_v1/download_benchmarks.py
```

Set Azure OpenAI environment variables:

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2025-01-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-5.4"
```

Run a quick benchmark smoke test:

```bash
./ddrg_v1/.venv/bin/python ./ddrg_v1/run_v1.py \
  --benchmark logiqa \
  --limit 1 \
  --llm-provider azure \
  --model gpt-5.4 \
  --k 1 \
  --max-workers 1 \
  --max-probes 0 \
  --print-each
```

## Results

Pilot experiment outputs are under `ddrg_v1/results/pilot_20260514/`.

Useful files:

- `summary.md`: readable benchmark summary
- `summary.csv`: machine-readable benchmark summary
- `per_example.csv`: per-example breakdown

For `medqa`, use `medqa_retry.jsonl` as the valid pilot output. The earlier `medqa.jsonl` file is a partial run captured before per-example error handling was added.

See `ddrg_v1/README.md` for the method-specific usage notes.

## Research Plan

For the current project framing, completed learned-anchor milestone, and next
research/development steps, see [docs/research_plan.md](./docs/research_plan.md).
