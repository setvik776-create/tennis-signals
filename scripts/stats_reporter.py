#!/usr/bin/env python3
"""Send prediction accuracy report and tracker CSV to Telegram."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path


BASE_DIR = Path("/root/tennis_signals")
DEFAULT_ENV = BASE_DIR / ".env"
TRACKER_FILE = BASE_DIR / "data" / "prediction_tracker.csv"
LOG_FILE = BASE_DIR / "logs" / "stats_reporter.log"
TRACKER_COLUMNS = [
    "run_at",
    "target",
    "match_date",
    "tournament",
    "surface",
    "player1",
    "player2",
    "predicted_winner",
    "player1_win_probability",
    "player2_win_probability",
    "confidence",
    "model_mode",
    "actual_winner",
    "is_correct",
]


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("\"'")
    return env


def ensure_tracker(path: Path):
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=TRACKER_COLUMNS).to_csv(path, index=False)
    return pd.read_csv(path)


def bool_series(values):
    return values.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def build_report(df) -> str:
    yesterday = date.today() - timedelta(days=1)
    resolved = df[df.get("actual_winner", "").fillna("").astype(str).ne("")].copy() if "actual_winner" in df else df.iloc[0:0].copy()
    total = len(resolved)
    correct = int(bool_series(resolved["is_correct"]).sum()) if total and "is_correct" in resolved else 0
    accuracy = (correct / total * 100.0) if total else 0.0

    if "match_date" in resolved:
        match_dates = pd_to_dates(resolved["match_date"])
        yesterday_df = resolved[match_dates.eq(yesterday)]
    else:
        yesterday_df = resolved.iloc[0:0]
    y_total = len(yesterday_df)
    y_correct = int(bool_series(yesterday_df["is_correct"]).sum()) if y_total and "is_correct" in yesterday_df else 0
    y_accuracy = (y_correct / y_total * 100.0) if y_total else 0.0

    return (
        "📊 *Tennis Oracle Stats Report*\n\n"
        f"Bendras tikslumas: *{accuracy:.1f}%* ({correct}/{total})\n"
        f"Vakar dienos pataikymai: *{y_correct}/{y_total}* ({y_accuracy:.1f}%)\n"
        f"Tracker eiluciu: *{len(df)}*"
    )


def pd_to_dates(values):
    import pandas as pd

    raw = values.fillna("").astype(str).str.strip()
    ymd = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    generic = pd.to_datetime(raw, errors="coerce")
    return ymd.fillna(generic).dt.date


def send_message(token: str, chat_id: str, text: str) -> tuple[int, str]:
    import requests

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True},
        timeout=30,
    )
    if response.status_code >= 300:
        raise RuntimeError(f"sendMessage HTTP {response.status_code}: {response.text}")
    return response.status_code, response.text


def send_document(token: str, chat_id: str, path: Path) -> tuple[int, str]:
    import requests

    with path.open("rb") as file_obj:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendDocument",
            data={"chat_id": chat_id, "caption": "prediction_tracker.csv"},
            files={"document": (path.name, file_obj, "text/csv")},
            timeout=30,
        )
    if response.status_code >= 300:
        raise RuntimeError(f"sendDocument HTTP {response.status_code}: {response.text}")
    return response.status_code, response.text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report prediction accuracy to Telegram.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--tracker", type=Path, default=TRACKER_FILE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    try:
        env = {**load_env(args.env_file), **os.environ}
        token = env.get("TELEGRAM_BOT_TOKEN")
        chat_id = env.get("ADMIN_TELEGRAM_CHAT_ID") or env.get("TELEGRAM_ADMIN_CHAT_ID") or env.get("TELEGRAM_CHAT_ID")
        df = ensure_tracker(args.tracker)
        report = build_report(df)
        if args.dry_run:
            print(report)
            print(f"CSV attachment: {args.tracker}")
            return 0
        if not token or not chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN and admin chat id/TELEGRAM_CHAT_ID in .env")
        msg_status, _ = send_message(token, chat_id, report)
        doc_status, _ = send_document(token, chat_id, args.tracker)
        logging.info("Stats report sent: sendMessage=%s sendDocument=%s chat_id=%s", msg_status, doc_status, chat_id)
        print(f"Telegram sendMessage HTTP {msg_status} OK")
        print(f"Telegram sendDocument HTTP {doc_status} OK")
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
