#!/usr/bin/env python3
"""Find weak tennis prediction rows that need player-data enrichment.

This first working version is intentionally conservative:
- reads predictions/history CSV files;
- writes candidates and staging rows;
- does not modify the main historical database.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from model_router import ModelRouter, write_probe


DEFAULT_HOST_DATA_DIR = Path("/root/tennis_signals/data")
DEFAULT_HOST_LOG_DIR = Path("/root/tennis_signals/logs")
DATA_DIR = Path(os.getenv("TENNIS_DATA_DIR") or (DEFAULT_HOST_DATA_DIR if DEFAULT_HOST_DATA_DIR.exists() else "/data"))
LOG_DIR = Path(os.getenv("TENNIS_LOG_DIR") or (DEFAULT_HOST_LOG_DIR if DEFAULT_HOST_LOG_DIR.exists() else "/logs"))

PREDICTIONS = DATA_DIR / "predictions.csv"
LOW_CONFIDENCE = DATA_DIR / "low_confidence_predictions.csv"
HISTORY = DATA_DIR / "tennis_all_matches_2024_to_now.csv"
CANDIDATES = DATA_DIR / "enrichment_candidates.csv"
STAGING = DATA_DIR / "player_enrichment_staging.csv"
REPORT = DATA_DIR / "enrichment_report.txt"
LOG_FILE = LOG_DIR / "enrichment.log"
MODEL_PROBE = DATA_DIR / "enrichment_model_probe.json"

CANDIDATE_COLUMNS = [
    "created_at",
    "match_date",
    "tournament",
    "surface",
    "player",
    "opponent",
    "confidence",
    "predicted_winner",
    "normalized_player",
    "history_matches",
    "reasons",
]

STAGING_COLUMNS = [
    "created_at",
    "status",
    "player",
    "normalized_player",
    "history_matches",
    "source_prediction_date",
    "source_tournament",
    "source_surface",
    "source_confidence",
    "reasons",
    "notes",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{now_iso()} {message}"
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_text = ascii_text.replace("-", " ")
    ascii_text = re.sub(r"[^a-zA-Z\s.]", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


def normalized_player(name: str) -> str:
    text = fold_text(name)
    return re.sub(r"\b([a-z])\.\s*", r"\1 ", text).strip()


def looks_initialed(name: str) -> bool:
    return bool(re.match(r"^\s*[A-ZÀ-Ž]\.", name or ""))


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def history_player_counts(history_rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in history_rows:
        for key in ("player1", "player2", "winner", "loser"):
            player = normalized_player(row.get(key, ""))
            if player:
                counts[player] += 1
    return counts


def prediction_rows() -> list[dict[str, str]]:
    low_rows = read_csv(LOW_CONFIDENCE)
    if low_rows:
        return low_rows
    rows = read_csv(PREDICTIONS)
    min_confidence = parse_float(os.getenv("MIN_CONFIDENCE", "0.55"), 0.55)
    return [row for row in rows if parse_float(row.get("confidence", "0")) < min_confidence]


def candidate_reasons(row: dict[str, str], player: str, counts: Counter[str]) -> list[str]:
    reasons: list[str] = []
    min_confidence = parse_float(os.getenv("MIN_CONFIDENCE", "0.55"), 0.55)
    low_history_threshold = int(os.getenv("HISTORY_LOW_MATCH_THRESHOLD", "3"))
    confidence = parse_float(row.get("confidence", "0"))
    surface = (row.get("surface") or "").strip().lower()
    norm = normalized_player(player)
    history_matches = counts.get(norm, 0)

    if confidence < min_confidence:
        reasons.append("low_confidence")
    if surface in {"", "unknown"}:
        reasons.append("unknown_surface")
    if looks_initialed(player):
        reasons.append("initialed_name")
    if history_matches == 0:
        reasons.append("not_found_in_history")
    elif history_matches < low_history_threshold:
        reasons.append("low_history_count")
    return reasons


def build_candidates(rows: list[dict[str, str]], counts: Counter[str]) -> list[dict[str, object]]:
    created_at = now_iso()
    candidates: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        players = [
            (row.get("player1", "").strip(), row.get("player2", "").strip()),
            (row.get("player2", "").strip(), row.get("player1", "").strip()),
        ]
        for player, opponent in players:
            if not player:
                continue
            reasons = candidate_reasons(row, player, counts)
            if not reasons:
                continue
            norm = normalized_player(player)
            key = (row.get("match_date", ""), norm, row.get("tournament", ""))
            candidates[key] = {
                "created_at": created_at,
                "match_date": row.get("match_date", ""),
                "tournament": row.get("tournament", ""),
                "surface": row.get("surface", ""),
                "player": player,
                "opponent": opponent,
                "confidence": row.get("confidence", ""),
                "predicted_winner": row.get("predicted_winner", ""),
                "normalized_player": norm,
                "history_matches": counts.get(norm, 0),
                "reasons": ";".join(reasons),
            }
    return sorted(candidates.values(), key=lambda item: (str(item["match_date"]), str(item["player"])))


def build_staging(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        norm = str(candidate["normalized_player"])
        existing = rows.get(norm)
        history_matches = int(candidate.get("history_matches") or 0)
        row = {
            "created_at": candidate["created_at"],
            "status": "needs_enrichment",
            "player": candidate["player"],
            "normalized_player": norm,
            "history_matches": history_matches,
            "source_prediction_date": candidate["match_date"],
            "source_tournament": candidate["tournament"],
            "source_surface": candidate["surface"],
            "source_confidence": candidate["confidence"],
            "reasons": candidate["reasons"],
            "notes": "Do not merge into main history before external source validation.",
        }
        if existing is None or history_matches < int(existing.get("history_matches") or 0):
            rows[norm] = row
    return sorted(rows.values(), key=lambda item: (int(item["history_matches"]), str(item["player"])))


def write_report(candidates: list[dict[str, object]], staging: list[dict[str, object]]) -> None:
    reason_counts: Counter[str] = Counter()
    for candidate in candidates:
        for reason in str(candidate.get("reasons", "")).split(";"):
            if reason:
                reason_counts[reason] += 1

    lines = [
        "Tennis enrichment report",
        f"Generated: {now_iso()}",
        f"Candidate rows: {len(candidates)}",
        f"Unique staging players: {len(staging)}",
        "",
        "Reason counts:",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            f"Candidates CSV: {CANDIDATES}",
            f"Staging CSV: {STAGING}",
            "Main history was not modified.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    try:
        log("Enrichment agent started")
        predictions = prediction_rows()
        history_rows = read_csv(HISTORY)
        counts = history_player_counts(history_rows)
        candidates = build_candidates(predictions, counts)
        staging = build_staging(candidates)
        write_csv(CANDIDATES, candidates, CANDIDATE_COLUMNS)
        write_csv(STAGING, staging, STAGING_COLUMNS)
        write_report(candidates, staging)
        if os.getenv("ENABLE_MODEL_ROUTER", "false").lower() in {"1", "true", "yes"}:
            try:
                router = ModelRouter()
                probe = router.daily_probe(len(candidates), len(staging))
                write_probe(MODEL_PROBE, probe)
                log(f"Model router probe OK: model={probe['model']} status={probe['status']}")
            except Exception as exc:
                log(f"Model router probe FAILED: {exc}")
        log(
            "Enrichment agent finished: "
            f"prediction_rows={len(predictions)} candidates={len(candidates)} staging_players={len(staging)}"
        )
        print(f"Candidates CSV: {CANDIDATES} rows={len(candidates)}")
        print(f"Staging CSV: {STAGING} rows={len(staging)}")
        print(f"Report: {REPORT}")
        print("Main history modified: no")
        return 0
    except Exception as exc:
        log(f"ERROR {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
