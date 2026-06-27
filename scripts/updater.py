#!/usr/bin/env python3
"""Fetch yesterday's ESPN tennis results and update history/tracker files."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any


BASE_DIR = Path("/root/tennis_signals")
HISTORY_FILE = BASE_DIR / "data" / "tennis_all_matches_2024_to_now.csv"
TRACKER_FILE = BASE_DIR / "data" / "prediction_tracker.csv"
LOG_FILE = BASE_DIR / "logs" / "updater.log"
ESPN_SCOREBOARD_URL = "https://www.espn.com/tennis/scoreboard/_/date/{date_key}"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )


def looks_like_player(line: str) -> bool:
    line = line.strip()
    if not line or line.isdigit():
        return False
    if re.match(r"^\d{1,2}:\d{2}\s?[AP]M(?:,\s*\d{1,2}/\d{1,2})?$", line, re.I):
        return False
    if re.search(r"\b(Set|Final|Retired|Walkover|Suspended|Postponed)\b", line, re.I):
        return False
    if re.search(r"\b(Round|Court|Centre|Center|Semifinal|Quarterfinal|Qualifier)\b", line, re.I):
        return False
    return bool(re.search(r"[A-Za-zÀ-ž]", line))


def is_completed(text: str) -> bool:
    return bool(re.search(r"\b(Final|Walkover|Retired)\b", text, re.I))


def parse_scores(lines: list[str], first_player: str, second_player: str) -> tuple[list[int], list[int]]:
    first_idx = lines.index(first_player)
    second_idx = lines.index(second_player)
    first_scores = [int(item) for item in lines[first_idx + 1 : second_idx] if item.isdigit()]
    second_scores: list[int] = []
    for item in lines[second_idx + 1 :]:
        if re.search(r"\b(Court|Centre|Center|Semifinal|Final|Round)\b", item, re.I):
            break
        if item.isdigit():
            second_scores.append(int(item))
    return first_scores, second_scores


def infer_winner(text: str) -> tuple[str, str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    players = [line for line in lines if looks_like_player(line)]
    if len(players) < 2:
        return None
    player1, player2 = players[0], players[1]
    if player1 == player2:
        return None
    try:
        p1_scores, p2_scores = parse_scores(lines, player1, player2)
    except Exception:
        p1_scores, p2_scores = [], []
    p1_sets = sum(1 for p1, p2 in zip(p1_scores, p2_scores) if p1 > p2)
    p2_sets = sum(1 for p1, p2 in zip(p1_scores, p2_scores) if p2 > p1)
    if p2_sets > p1_sets:
        return player2, player1
    return player1, player2


def infer_surface(tournament: str) -> str:
    text = tournament.lower()
    if any(hint in text for hint in ["wimbledon", "eastbourne", "homburg", "mallorca", "halle", "queen"]):
        return "Grass"
    if any(hint in text for hint in ["roland", "monte carlo", "madrid", "rome", "barcelona", "gstaad", "bastad", "umag"]):
        return "Clay"
    if any(hint in text for hint in ["australian", "us open", "miami", "indian wells", "cincinnati", "toronto", "montreal"]):
        return "Hard"
    return "Unknown"


async def fetch_espn_results(day: date) -> list[dict[str, str]]:
    from playwright.async_api import async_playwright

    url = ESPN_SCOREBOARD_URL.format(date_key=day.strftime("%Y%m%d"))
    logging.info("Opening ESPN results page in headless Chromium: %s", url)
    results: list[dict[str, str]] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=USER_AGENT)
        response = await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        logging.info("ESPN response status: %s", response.status if response else "unknown")
        try:
            await page.wait_for_selector("section.Card", timeout=20000)
        except Exception as exc:
            logging.warning("ESPN cards did not load: %s", exc)
        await page.wait_for_timeout(3000)
        cards: list[dict[str, Any]] = await page.locator("section.Card").evaluate_all(
            """cards => cards.map(card => ({
                tournament: (card.querySelector('.Tournament_Header')?.innerText || '').trim(),
                groups: Array.from(card.querySelectorAll('.Grouping')).map(group => ({
                    heading: (group.firstElementChild?.innerText || '').trim(),
                    competitions: Array.from(group.querySelectorAll('.CompetitionsWrapper > div')).map(comp => comp.innerText)
                }))
            }))"""
        )
        await browser.close()

    for card in cards:
        tournament = str(card.get("tournament") or "").strip()
        if not tournament:
            continue
        for group in card.get("groups") or []:
            heading = str(group.get("heading") or "").upper()
            if "SINGLES" not in heading or "DOUBLES" in heading:
                continue
            for competition in group.get("competitions") or []:
                text = str(competition)
                if not is_completed(text):
                    continue
                parsed = infer_winner(text)
                if not parsed:
                    continue
                winner, loser = parsed
                results.append(
                    {
                        "date": day.strftime("%Y%m%d"),
                        "tournament": tournament,
                        "tournament_level": "",
                        "surface": infer_surface(tournament),
                        "round": "",
                        "winner": winner,
                        "loser": loser,
                        "score": "",
                        "best_of": "",
                        "winner_rank": "",
                        "loser_rank": "",
                        "minutes": "",
                        "tour": "ATP/WTA",
                        "aces": "",
                        "double_faults": "",
                        "first_serve_pct": "",
                        "first_serve_points_won": "",
                        "second_serve_points_won": "",
                        "break_points_saved": "",
                        "break_points_converted": "",
                        "total_points_won": "Unknown",
                    }
                )
    return results


def append_history(results: list[dict[str, str]], history_file: Path) -> int:
    if not results:
        return 0
    import pandas as pd

    new_df = pd.DataFrame(results)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    if history_file.exists():
        history = pd.read_csv(history_file)
        combined = pd.concat([history, new_df], ignore_index=True)
    else:
        combined = new_df
    before = len(combined)
    combined = combined.drop_duplicates(subset=["date", "winner", "loser", "tournament"], keep="last")
    combined.to_csv(history_file, index=False)
    return before - len(combined)


def update_tracker(results: list[dict[str, str]], tracker_file: Path, day: date) -> int:
    if not tracker_file.exists() or not results:
        return 0
    import pandas as pd

    tracker = pd.read_csv(tracker_file)
    if tracker.empty:
        return 0
    for column in ["actual_winner", "is_correct"]:
        if column not in tracker.columns:
            tracker[column] = ""
        tracker[column] = tracker[column].fillna("").astype("object")

    result_map = {"||".join(sorted([row["winner"], row["loser"]])): row["winner"] for row in results}
    tracker_dates = pd.to_datetime(tracker["match_date"], errors="coerce").dt.date
    updated = 0
    for idx, row in tracker[tracker_dates.eq(day)].iterrows():
        key = "||".join(sorted([str(row["player1"]), str(row["player2"])]))
        winner = result_map.get(key)
        if not winner:
            continue
        tracker.loc[idx, "actual_winner"] = winner
        tracker.loc[idx, "is_correct"] = str(winner == row["predicted_winner"])
        updated += 1
    tracker.to_csv(tracker_file, index=False)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update history and tracker from ESPN tennis results.")
    parser.add_argument("--history", type=Path, default=HISTORY_FILE)
    parser.add_argument("--tracker", type=Path, default=TRACKER_FILE)
    parser.add_argument("--date", default=(date.today() - timedelta(days=1)).isoformat())
    return parser.parse_args()


async def async_main() -> int:
    setup_logging()
    args = parse_args()
    try:
        day = date.fromisoformat(args.date)
        results = await fetch_espn_results(day)
        removed_dupes = append_history(results, args.history)
        tracker_updates = update_tracker(results, args.tracker, day)
        logging.info(
            "Updater finished: results=%d, duplicates_removed=%d, tracker_updates=%d",
            len(results),
            removed_dupes,
            tracker_updates,
        )
        print(f"Browser opened headless: OK")
        print(f"Result date: {day.isoformat()}")
        print(f"Completed singles results found: {len(results)}")
        print(f"Tracker rows updated: {tracker_updates}")
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))
