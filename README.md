# Legal Benchmark Dashboard

A leaderboard and interactive dashboard tracking LLM performance on legal reasoning tasks, powered by [LegalBench](https://huggingface.co/datasets/nguha/legalbench).

**Live dashboard:** [https://hazyresearch.github.io/legal-benchmark-dashboard](https://hazyresearch.github.io/legal-benchmark-dashboard)

## Quick Start

### View the dashboard locally

```bash
cd dashboard
python -m http.server 8080
# Open http://localhost:8080
```

### Run evaluations

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Quick test (5 samples, instant feedback)
python run.py test --models claude-haiku-4.5 --sample-size 100

# 4. Full evaluation via batch API
python run.py submit --models claude-haiku-4.5
python run.py status        # check progress
python run.py download      # download + score + export
```

## Running Evaluations

### Batch API (Anthropic, OpenAI, Google)

The `submit`/`status`/`download` workflow uses provider batch APIs for cost-efficient evaluation:

```bash
# Submit (creates batch job, returns immediately)
python run.py submit --models claude-sonnet-4.6 gpt-5-mini

# Check progress
python run.py status

# Download results, score, and export leaderboard
python run.py download

# Score (or re-score) existing raw results
python run.py score --models claude-sonnet-4.6
```

### Direct API (xAI, OpenRouter)

Some providers don't support batch APIs for all models. Use the standalone generate scripts, which run concurrent direct API calls:

```bash
# xAI (Grok models)
python generate_xai_predictions.py --models grok-4.2-non-reasoning grok-4.2-reasoning

# OpenRouter (open-weight and third-party models)
python generate_openrouter_predictions.py --models mimo-v2-pro deepseek-v3.2

# Google (Gemini models, if batch API is unavailable)
python generate_google_predictions.py --models gemini-3.1-pro

# Then score
python run.py score
```

### Test a new model

```bash
# Verify a model works before full evaluation
python run.py test-batch --models gpt-5-mini --size 5
```

### Clean up stale jobs

```bash
python run.py clean
```

## Adding Models

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
  effort: high                  # OpenAI reasoning effort (none/low/medium/high/xhigh)
  thinking_level: high          # Google Gemini thinking level (minimal/low/medium/high)
  reasoning: true               # OpenRouter: enable reasoning for supported models
```

## Supported Providers

| Provider | Env Variable | Batch API | Direct API |
|----------|-------------|-----------|------------|
| Anthropic | `ANTHROPIC_API_KEY` | Yes (`run.py submit`) | Yes (`run.py test`) |
| OpenAI | `OPENAI_API_KEY` | Yes (Responses API) | Yes |
| xAI | `XAI_API_KEY` | Limited | Yes (`generate_xai_predictions.py`) |
| Google | `GOOGLE_API_KEY` | Yes | Yes (`generate_google_predictions.py`) |
| OpenRouter | `OPENROUTER_API_KEY` | No | Yes (`generate_openrouter_predictions.py`) |

## Project Structure

```
legal-benchmark-dashboard/
├── dashboard/               # Static dashboard (HTML/CSS/JS)
│   ├── index.html
│   ├── css/style.css
│   ├── js/
│   │   ├── app.js           # Entry point, data loading
│   │   ├── leaderboard.js   # Main leaderboard table
│   │   └── taskview.js      # Per-task drill-down view
│   └── data/
│       └── leaderboard.json # Exported results (auto-generated)
├── providers/               # API provider implementations
│   ├── anthropic_provider.py
│   ├── openai_provider.py
│   ├── xai_provider.py
│   └── google_provider.py
├── benchmarks/
│   └── legalbench.py        # Task loading from HuggingFace
├── run.py                   # Main CLI (submit/status/download/score)
├── scoring.py               # Answer extraction and evaluation
├── export.py                # Export results to dashboard JSON
├── generate_xai_predictions.py
├── generate_openrouter_predictions.py
├── generate_google_predictions.py
├── models.yaml              # Model registry
├── results/                 # Raw and scored results
│   └── legalbench/
│       ├── raw/             # Raw model responses (JSONL)
│       └── *.json           # Scored results per model
└── .env                     # API keys (not committed)
```

## Hosting

The dashboard is a static site (no server required). To deploy:

### GitHub Pages

1. Push this repo to GitHub
2. Go to Settings → Pages → Source: "Deploy from a branch"
3. Select `main` branch, `/dashboard` folder
4. The dashboard will be live at `https://<org>.github.io/<repo>/`

### Local

```bash
cd dashboard && python -m http.server 8080
```

## Data Pipeline

```
LegalBench (HuggingFace)
    ↓  load_tasks()
models.yaml + .env
    ↓  run.py submit / generate_*.py
results/legalbench/raw/*.jsonl     ← raw model responses
    ↓  run.py score
results/legalbench/*.json          ← scored results per model
    ↓  export.py
dashboard/data/leaderboard.json    ← dashboard data
    ↓
dashboard/index.html               ← static site
```
