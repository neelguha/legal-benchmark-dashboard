# Legal Benchmark Dashboard

A leaderboard tracking LLM performance on academic legal reasoning benchmarks.

**Live site:** [https://neelguha.github.io/legal-benchmark-dashboard/](https://neelguha.github.io/legal-benchmark-dashboard/)

The leaderboard evaluates models across five legal benchmarks — LegalBench, BarExam (MBE), LEXam, HousingQA, and Legal Hallucinations — sourced from the [`nguha/legal-eval`](https://huggingface.co/datasets/nguha/legal-eval) HuggingFace dataset.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Generate predictions for a model
python generate.py --models claude-haiku-4.5

# 4. Score and export to the dashboard
python run.py score --models claude-haiku-4.5

# 5. Preview the dashboard locally
cd dashboard && python -m http.server 8080
# Open http://localhost:8080
```

## Running evaluations

Generation and scoring are two separate steps:

### `generate.py` — run a model on the benchmarks

All providers use the same unified script. It makes concurrent direct API calls, saves results incrementally, and resumes on re-run so interrupted jobs pick up where they left off.

```bash
# One model
python generate.py --models claude-haiku-4.5

# Multiple models
python generate.py --models claude-sonnet-4.6 gpt-5-mini gemini-3.1-pro

# All models in models.yaml
python generate.py --models all

# Limit to specific benchmarks or tasks
python generate.py --models claude-haiku-4.5 --benchmarks legalbench barexam
python generate.py --models claude-haiku-4.5 --tasks abercrombie hearsay mbe

# Force re-generation (ignore existing responses)
python generate.py --models claude-haiku-4.5 --benchmarks lexam --fresh

# Tune concurrency
python generate.py --models claude-haiku-4.5 --concurrency 20
```

Raw responses are saved to `results/legal-eval/raw/<model_key>.jsonl`.

### `run.py score` — score and export

Scoring reads the raw JSONL files, computes per-task and per-benchmark accuracy, and regenerates `dashboard/data/leaderboard.json`.

```bash
# Score all models with raw files
python run.py score

# Or just specific models
python run.py score --models claude-haiku-4.5 gpt-5-mini
```

Scored results go to `results/legal-eval/<model_key>.json`. The export runs automatically after scoring.

## Adding models

Add an entry to `models.yaml`:

```yaml
my-model:
  provider: openai              # anthropic, openai, xai, google, openrouter
  model_id: gpt-5-mini          # vendor's model ID
  display_name: GPT-5 Mini      # shown on leaderboard
  developer: OpenAI             # organization that built the model
  max_tokens: 4096              # max output tokens
  open_weight: false            # true for open-weight models

  # Optional provider-specific params:
  effort: high                  # OpenAI reasoning effort: low/medium/high
  thinking_level: high          # Google Gemini: minimal/low/medium/high
  reasoning: true               # OpenRouter: enable reasoning for supported models

  # Optional per-model system prompt override (otherwise uses the default in models.yaml)
  system_prompt: "Answer briefly."
```

Then:
```bash
python generate.py --models my-model
python run.py score --models my-model
```

## Supported providers

| Provider | Env Variable |
|----------|-------------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| xAI | `XAI_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

All providers are called via direct API (concurrent requests). No batch APIs.

## Project structure

```
legal-benchmark-dashboard/
├── dashboard/               # Static dashboard (HTML/CSS/JS)
│   ├── index.html           # Single-page leaderboard
│   └── data/
│       ├── leaderboard.json # Exported results (committed, auto-generated)
│       └── benchmarks.json  # Page content config (title, links)
├── benchmarks/
│   └── legal_eval.py        # Task loading from nguha/legal-eval
├── .github/workflows/
│   └── deploy.yml           # GitHub Pages deployment
├── generate.py              # Generation entry point (all providers)
├── run.py                   # Scoring and export
├── export.py                # Builds leaderboard.json from scored results
├── scoring.py               # Answer extraction and eval methods
├── models.yaml              # Model registry
├── config.py                # API key loading
├── results/                 # Raw and scored results (gitignored)
│   └── legal-eval/
│       ├── raw/             # Raw model responses (JSONL)
│       └── *.json           # Scored results per model
└── .env                     # API keys (not committed)
```

## Hosting

The dashboard is a static site (no server required). It's deployed to GitHub Pages via the workflow in `.github/workflows/deploy.yml`, which triggers on every push to `main`.

To enable Pages on a fork:

1. Push the repo to GitHub
2. Go to **Settings → Pages**
3. Under **Source**, select **GitHub Actions** (not "Deploy from a branch")
4. Push to `main` — the workflow deploys `dashboard/` as the site

### Local preview

```bash
cd dashboard && python -m http.server 8080
# Open http://localhost:8080
```

## Data pipeline

```
nguha/legal-eval (HuggingFace)
    ↓  benchmarks/legal_eval.py load_tasks()
models.yaml + .env
    ↓  python generate.py
results/legal-eval/raw/*.jsonl     ← raw model responses
    ↓  python run.py score
results/legal-eval/*.json          ← scored results per model
    ↓  export.py (automatic)
dashboard/data/leaderboard.json    ← dashboard data
    ↓  git push
GitHub Actions → GitHub Pages      ← live site
```

## Benchmarks

Details on each benchmark (sources, sampling, prompt construction) are in the dataset README at [`nguha/legal-eval`](https://huggingface.co/datasets/nguha/legal-eval). The site's About section also summarizes them.

## Maintenance

Maintained by [Neel Guha](https://neelguha.com). Reach out with questions, comments, or concerns.
