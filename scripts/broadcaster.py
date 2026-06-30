#!/usr/bin/env python3
"""Send tennis predictions to Telegram."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV = BASE_DIR / ".env"
DEFAULT_PREDICTIONS = BASE_DIR / "data" / "predictions.csv"
DEFAULT_LOW_CONFIDENCE = BASE_DIR / "data" / "low_confidence_predictions.csv"
LOG_FILE = BASE_DIR / "logs" / "broadcaster.log"


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


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


def confidence_label(value: float) -> str:
    if value >= 0.65:
        return "stiprus"
    if value >= 0.55:
        return "vidutinis"
    return "zemas"


def load_predictions(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install with: python3 -m pip install pandas") from exc

    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    df = pd.read_csv(path)
    required = {
        "match_date",
        "tournament",
        "surface",
        "player1",
        "player2",
        "predicted_winner",
        "player1_win_probability",
        "player2_win_probability",
        "confidence",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")
    df["confidence"] = df["confidence"].astype(float)
    return df


def write_low_confidence_csv(df, min_confidence: float, output: Path) -> int:
    low_df = df[df["confidence"] < min_confidence].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    low_df.to_csv(output, index=False)
    return len(low_df)


def build_message_from_df(df, min_confidence: float, limit: int) -> str:
    df = df[df["confidence"] >= min_confidence].copy()
    df = df.sort_values("confidence", ascending=False).head(limit)
    if df.empty:
        return f"Teniso prognozes: siandien nera signalu virs {format_percent(min_confidence)}."

    dates = ", ".join(sorted(str(item) for item in df["match_date"].dropna().unique()))
    lines = [
        "🎾 *Tennis Oracle Signal*",
        f"Data: {dates or 'Nenurodyta'}",
        f"Publikuojami signalai: {len(df)}",
        "",
    ]
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        p1_prob = float(row["player1_win_probability"])
        p2_prob = float(row["player2_win_probability"])
        winner_prob = p1_prob if row["predicted_winner"] == row["player1"] else p2_prob
        tournament = str(row.get("tournament", "")).strip()
        surface = str(row.get("surface", "")).strip()
        meta = " / ".join(item for item in [tournament, surface] if item)
        lines.append(
            f"*{index}. Rungtynes:* *{row['player1']}* vs *{row['player2']}*\n"
            f"Turnyras: {meta or 'Nenurodyta'}\n"
            f"Prognozuojamas laimetojas: *{row['predicted_winner']}*\n"
            f"Pasitikejimas: *{format_percent(float(row['confidence']))}* "
            f"({confidence_label(float(row['confidence']))}, win prob. {format_percent(winner_prob)})"
        )
        lines.append("")
    return "\n".join(lines)


def split_telegram_message(text: str, max_len: int = 3800) -> list[str]:
    chunks: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_len:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = block
    if current:
        chunks.append(current)
    return chunks


def send_telegram(token: str, chat_id: str, text: str) -> list[tuple[int, str]]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required. Install with: python3 -m pip install requests") from exc

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    responses: list[tuple[int, str]] = []
    chunks = split_telegram_message(text)
    for index, chunk in enumerate(chunks, start=1):
        suffix = f"\n\nDalis {index}/{len(chunks)}" if len(chunks) > 1 else ""
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": f"{chunk}{suffix}",
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=30,
        )
        if response.status_code >= 300:
            raise RuntimeError(f"Telegram API returned HTTP {response.status_code}: {response.text}")
        responses.append((response.status_code, response.text))
    return responses


def send_document(token: str, chat_id: str, path: Path, caption: str) -> tuple[int, str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required. Install with: python3 -m pip install requests") from exc

    with path.open("rb") as file_obj:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendDocument",
            data={"chat_id": chat_id, "caption": caption},
            files={"document": (path.name, file_obj, "text/csv")},
            timeout=30,
        )
    if response.status_code >= 300:
        raise RuntimeError(f"Telegram sendDocument returned HTTP {response.status_code}: {response.text}")
    return response.status_code, response.text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Broadcast tennis predictions to Telegram.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--chat-id", default=os.getenv("TELEGRAM_CHAT_ID"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--low-confidence-output", type=Path, default=DEFAULT_LOW_CONFIDENCE)
    parser.add_argument("--skip-low-confidence-document", action="store_true")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    try:
        env = load_env(args.env_file)
        token = args.token or env.get("TELEGRAM_BOT_TOKEN")
        chat_id = args.chat_id or env.get("TELEGRAM_CHAT_ID")
        predictions = load_predictions(args.predictions)
        low_count = write_low_confidence_csv(predictions, args.min_confidence, args.low_confidence_output)
        high_count = int((predictions["confidence"] >= args.min_confidence).sum())
        message = build_message_from_df(predictions, args.min_confidence, args.limit)
        if args.dry_run:
            print(message)
            print(f"\nSignals >= {format_percent(args.min_confidence)}: {high_count}")
            print(f"Signals < {format_percent(args.min_confidence)} CSV: {args.low_confidence_output} rows={low_count}")
            return 0
        if not token or not chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID or --token/--chat-id.")
        responses = send_telegram(token, chat_id, message)
        document_response = None
        if low_count and not args.skip_low_confidence_document:
            document_response = send_document(
                token,
                chat_id,
                args.low_confidence_output,
                f"Zemo pasitikejimo signalai (< {format_percent(args.min_confidence)})",
            )
        for status_code, response_text in responses:
            logging.info("Telegram API HTTP %s response: %s", status_code, response_text)
        if document_response:
            logging.info("Telegram sendDocument HTTP %s response: %s", document_response[0], document_response[1])
        print(f"Telegram API HTTP {responses[-1][0]} OK messages_sent={len(responses)}")
        if document_response:
            print(f"Telegram sendDocument HTTP {document_response[0]} OK low_confidence_rows={low_count}")
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
