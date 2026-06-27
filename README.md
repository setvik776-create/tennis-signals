# Tennis Signals

Automated tennis match signal pipeline with daily scraping, prediction scoring, Telegram broadcasting, result tracking, and a conservative enrichment agent.

## What It Does

- Scrapes target-day tennis matches from ESPN and Tennis.com.
- Trains a local model from historical match data.
- Writes prediction CSV files and tracks outcomes.
- Sends high-confidence signals and reports to Telegram.
- Runs a Dockerized enrichment agent that identifies weak player-data rows for staging.
- Uses a model router probe for Gemini-compatible enrichment checks when enabled.

## Main Commands

```bash
./run_pipeline.sh today
./run_pipeline.sh tomorrow
./run_enrichment_agent.sh
```

## Required Secrets

Do not commit real secrets. Keep them in `.env` files.

Root `.env`:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ADMIN_TELEGRAM_CHAT_ID=
```

Enrichment agent `.env`:

```env
GOOGLE_GEMINI_API_KEY=
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
ENABLE_MODEL_ROUTER=true
```

See `enrichment_agent/.env.example` for model routing options.

## Data

Runtime files under `data/` and `logs/` are intentionally ignored by git. The production machine should keep its private historical CSV files, generated predictions, model artefacts, reports, and logs locally.
