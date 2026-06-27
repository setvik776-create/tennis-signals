# Tennis Signals

Tennis Signals is an automated tennis prediction system. It scrapes upcoming matches, trains a local model from historical results, generates match predictions, sends Telegram signals, tracks outcomes, and stages weak player-data cases for enrichment.

The repository contains the runnable project code plus the current data/model artefacts needed to reproduce the production state. Real `.env` files and logs are intentionally not committed.

## Current Capabilities

- Scrapes singles matches for `today` or `tomorrow`.
- Combines ESPN and Tennis.com match sources.
- Infers tournament surface when possible.
- Trains a local ML model from historical tennis results.
- Produces prediction CSVs with winner probabilities and confidence.
- Sends Telegram messages for high-confidence signals.
- Sends low-confidence predictions as a CSV document.
- Tracks predictions and later updates actual winners.
- Sends a daily stats report to Telegram.
- Runs a Dockerized enrichment agent for weak player-data detection.
- Uses a model router probe for Gemini-compatible enrichment checks when enabled.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── setup_env.py
├── run_pipeline.sh
├── run_enrichment_agent.sh
├── run_opencode_gemma.sh
├── data/
│   ├── tennis_all_matches_2024_to_now.csv
│   ├── tennis_model.joblib
│   ├── target_matches.csv
│   ├── predictions.csv
│   ├── low_confidence_predictions.csv
│   ├── prediction_tracker.csv
│   ├── enrichment_candidates.csv
│   ├── player_enrichment_staging.csv
│   ├── enrichment_model_probe.json
│   └── enrichment_report.txt
├── scripts/
│   ├── scraper.py
│   ├── predictor.py
│   ├── broadcaster.py
│   ├── updater.py
│   └── stats_reporter.py
└── enrichment_agent/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── enrichment_agent.py
    ├── model_router.py
    └── .env.example
```

## Data Files

The `data/` directory is committed so the repo contains the current project state:

- `tennis_all_matches_2024_to_now.csv` - historical match database.
- `tennis_model.joblib` - trained local model artefact.
- `target_matches.csv` - latest scraped target matches.
- `predictions.csv` - latest generated predictions.
- `low_confidence_predictions.csv` - low-confidence prediction export.
- `prediction_tracker.csv` - prediction history and result tracking.
- `enrichment_candidates.csv` - player rows selected for enrichment review.
- `player_enrichment_staging.csv` - conservative staging table for enrichment.
- `enrichment_model_probe.json` - latest model-router probe result.
- `enrichment_report.txt` - latest enrichment summary.

## What Is Not Committed

These files stay local and should not be published:

- `.env`
- `enrichment_agent/.env`
- `.venv/`
- `__pycache__/`
- `logs/`
- `*.log`

Real API keys, Telegram bot tokens, chat IDs, and runtime logs should remain on the production machine.

## Requirements

- Python 3.12+
- Docker and Docker Compose for the enrichment agent
- Chromium dependencies for Playwright
- Telegram bot token for broadcasting
- Gemini/OpenAI-compatible key for enrichment model probing

Install Python dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Environment Setup

Copy the example file and fill in real values:

```bash
cp .env.example .env
chmod 600 .env
```

Root `.env`:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ADMIN_TELEGRAM_CHAT_ID=
```

The helper script can resolve a Telegram channel ID from `TELEGRAM_CHANNEL`:

```bash
export TELEGRAM_BOT_TOKEN="your-token"
export TELEGRAM_CHANNEL="@tennisoraclesignal"
python setup_env.py
```

Enrichment agent setup:

```bash
cp enrichment_agent/.env.example enrichment_agent/.env
chmod 600 enrichment_agent/.env
```

Key enrichment values:

```env
GOOGLE_GEMINI_API_KEY=
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GEMMA_PRIMARY_MODEL=gemma-4-31b-it
GEMMA_SECONDARY_MODEL=gemma-4-24b-a4b-it
GEMINI_LIGHT_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDING_MODEL_1=gemini-embedding-001
GEMINI_EMBEDDING_MODEL_2=gemini-embedding-002
ENABLE_MODEL_ROUTER=true
MIN_CONFIDENCE=0.55
HISTORY_LOW_MATCH_THRESHOLD=3
```

## Main Commands

Run the full prediction pipeline for today:

```bash
./run_pipeline.sh today
```

Run the full prediction pipeline for tomorrow:

```bash
./run_pipeline.sh tomorrow
```

Run the enrichment agent:

```bash
./run_enrichment_agent.sh
```

Run the result updater manually:

```bash
.venv/bin/python scripts/updater.py
```

Run the stats reporter manually:

```bash
.venv/bin/python scripts/stats_reporter.py
```

## Pipeline Flow

1. `scripts/scraper.py` collects target matches.
2. `scripts/predictor.py` trains on historical data and writes predictions.
3. `run_pipeline.sh` appends the run to `prediction_tracker.csv`.
4. `scripts/broadcaster.py` sends Telegram signals and low-confidence CSV.
5. `scripts/updater.py` checks completed matches and updates actual results.
6. `scripts/stats_reporter.py` sends aggregate accuracy stats.
7. `enrichment_agent/enrichment_agent.py` builds enrichment candidates and staging rows.

## Suggested Cron

Production cron used by this system:

```cron
0 6 * * * /root/tennis_signals/run_pipeline.sh today >> /root/tennis_signals/logs/cron_morning.log 2>&1
0 20 * * * /root/tennis_signals/run_pipeline.sh tomorrow >> /root/tennis_signals/logs/cron_evening.log 2>&1
0 3 * * * /root/tennis_signals/.venv/bin/python /root/tennis_signals/scripts/updater.py >> /root/tennis_signals/logs/updater.log 2>&1
0 21 * * * /root/tennis_signals/.venv/bin/python /root/tennis_signals/scripts/stats_reporter.py >> /root/tennis_signals/logs/stats_reporter.log 2>&1
30 22 * * * /root/tennis_signals/run_enrichment_agent.sh >> /root/tennis_signals/logs/enrichment_cron.log 2>&1
```

## Enrichment Agent

The enrichment agent is intentionally conservative. It reads the latest prediction and history files, then selects player rows that need more data because of:

- low prediction confidence;
- unknown surface;
- initialed names such as `A. Wolf`;
- players missing from history;
- very low historical match count.

It writes:

- `data/enrichment_candidates.csv`
- `data/player_enrichment_staging.csv`
- `data/enrichment_report.txt`
- `data/enrichment_model_probe.json`

It does not merge enrichment rows into the main historical database. External validation should happen before any main-history update.

## Model Router

`enrichment_agent/model_router.py` selects model IDs by task:

- `daily_probe`, `clean_scrape_text`, `structure_text` -> light Gemini model.
- `name_embedding`, `player_similarity`, `dedupe` -> embedding model.
- `resolve_player_match`, `uncertain_alias` -> secondary Gemma model.
- high ambiguity or audit decisions -> primary Gemma model.

The current probe uses a small chat-completions request and writes the result to `data/enrichment_model_probe.json`.

## Safety Notes

- Never commit real `.env` files.
- Rotate tokens if they were pasted into chat or logs.
- Review `data/` before publishing if future files may contain private information.
- Keep logs private because external API responses can contain operational details.
