#!/usr/bin/env python3
"""Collect target-day singles tennis matches from ESPN and Tennis.com."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any


BASE_DIR = Path("/root/tennis_signals")
DEFAULT_OUTPUT = BASE_DIR / "data" / "target_matches.csv"
LOG_FILE = BASE_DIR / "logs" / "scraper.log"
MATCH_COLUMNS = ["match_date", "tournament", "surface", "player1", "player2"]
ESPN_SCOREBOARD_URL = "https://www.espn.com/tennis/scoreboard/_/date/{date_key}"
TENNIS_COM_URL = "https://www.tennis.com/?date={date_key}&tour=all"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )


def target_date(target: str) -> date:
    today = date.today()
    return today if target == "today" else today + timedelta(days=1)


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


def parse_competition_text(text: str) -> tuple[str, str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    players = [line for line in lines if looks_like_player(line)]
    if len(players) < 2:
        return None
    player1, player2 = players[0], players[1]
    if player1 == player2:
        return None
    return player1, player2


def infer_surface(tournament: str) -> str:
    text = tournament.lower()
    grass_hint = ["wimbledon", "eastbourne", "homburg", "mallorca", "halle", "queen"]
    clay_hint = ["roland", "monte carlo", "madrid", "rome", "barcelona", "gstaad", "bastad", "umag"]
    hard_hint = ["australian", "us open", "miami", "indian wells", "cincinnati", "toronto", "montreal"]
    if any(hint in text for hint in grass_hint):
        return "Grass"
    if any(hint in text for hint in clay_hint):
        return "Clay"
    if any(hint in text for hint in hard_hint):
        return "Hard"
    return "Unknown"


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


def player_family_key(name: str) -> str:
    text = fold_text(name)
    text = re.sub(r"\b[a-z]\.\s*", "", text)
    text = re.sub(r"[^a-z\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_key(match: dict[str, str]) -> tuple[str, str, str]:
    p1 = player_family_key(match["player1"])
    p2 = player_family_key(match["player2"])
    ordered = sorted([p1, p2])
    return match["match_date"], ordered[0], ordered[1]


def same_player_key(left: str, right: str) -> bool:
    left_key = player_family_key(left)
    right_key = player_family_key(right)
    if not left_key or not right_key:
        return False
    return left_key == right_key or left_key in right_key or right_key in left_key


def same_match(left: dict[str, str], right: dict[str, str]) -> bool:
    if left["match_date"] != right["match_date"]:
        return False
    direct = same_player_key(left["player1"], right["player1"]) and same_player_key(left["player2"], right["player2"])
    swapped = same_player_key(left["player1"], right["player2"]) and same_player_key(left["player2"], right["player1"])
    return direct or swapped


def is_tennis_com_tournament(line: str) -> bool:
    if re.match(r"^UTR\b", line):
        return True
    if re.match(r"^ATP Challenger\b", line):
        return True
    if not bool(re.match(r"^(ATP|WTA|ATP/WTA)\b", line)):
        return False
    if "Singles" in line or "Doubles" in line:
        return False
    if line == "ATP & WTA":
        return False
    if re.match(r"^(ATP|WTA)\s+\d", line):
        return False
    if re.match(r"^WTA\s+\d+\s*/\s*ATP\s+\d+", line):
        return False
    return True


def is_tennis_com_event(line: str) -> bool:
    return "Singles" in line or "Doubles" in line


def looks_like_tennis_com_player(line: str) -> bool:
    line = line.strip()
    blocked = {
        "·",
        "Watch",
        "UPCOMING",
        "LIVE",
        "COMPLETED",
        "TBD",
        "Q",
        "Final",
        "Semifinal",
        "Quarterfinal",
        "Round of 16",
        "Projected Winner",
        "CHALLENGER",
        "Walkover",
    }
    if not line or line in blocked:
        return False
    if line.isdigit():
        return False
    if re.match(r"^\d+%$", line):
        return False
    if re.search(r"\b(today|tomorrow|your time|am|pm)\b", line, re.I):
        return False
    if is_tennis_com_event(line) or is_tennis_com_tournament(line):
        return False
    if line.isupper() and len(line) > 2:
        return False
    return bool(re.search(r"[A-Za-zÀ-ž]", line))


def next_tennis_com_boundary(lines: list[str], start: int) -> int:
    footer = {"COMPANY", "CAREERS", "CONTACT US", "MORE"}
    idx = start
    while idx < len(lines):
        line = lines[idx]
        if line in footer or is_tennis_com_event(line) or is_tennis_com_tournament(line):
            break
        idx += 1
    return idx


def parse_tennis_com_text(text: str, day: date) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    matches: list[dict[str, str]] = []
    tournament = ""
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if is_tennis_com_tournament(line):
            tournament = line
            if idx + 1 < len(lines):
                candidate = lines[idx + 1]
                if (
                    candidate not in {"·", "CHALLENGER"}
                    and not candidate.isupper()
                    and not is_tennis_com_event(candidate)
                    and not is_tennis_com_tournament(candidate)
                ):
                    tournament = candidate
                    idx += 1
            idx += 1
            continue

        if "Singles" in line and "Doubles" not in line:
            end = next_tennis_com_boundary(lines, idx + 1)
            block = lines[idx + 1 : end]
            if "COMPLETED" in block or "Walkover" in block:
                idx = end
                continue
            players = [candidate for candidate in block if looks_like_tennis_com_player(candidate)]
            if len(players) >= 2 and players[0] != players[1]:
                matches.append(
                    {
                        "match_date": day.isoformat(),
                        "tournament": tournament or "Tennis.com",
                        "surface": infer_surface(tournament),
                        "player1": players[0],
                        "player2": players[1],
                    }
                )
            idx = end
            continue
        idx += 1
    return matches


async def fetch_espn_matches(day: date) -> list[dict[str, str]]:
    from playwright.async_api import async_playwright

    url = ESPN_SCOREBOARD_URL.format(date_key=day.strftime("%Y%m%d"))
    logging.info("Opening ESPN scoreboard in headless Chromium: %s", url)
    matches: list[dict[str, str]] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=USER_AGENT)
        response = await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        status = response.status if response else "unknown"
        logging.info("ESPN response status: %s", status)
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
                parsed = parse_competition_text(str(competition))
                if not parsed:
                    continue
                player1, player2 = parsed
                matches.append(
                    {
                        "match_date": day.isoformat(),
                        "tournament": tournament,
                        "surface": infer_surface(tournament),
                        "player1": player1,
                        "player2": player2,
                    }
                )

    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for match in matches:
        key = (match["match_date"], match["player1"], match["player2"])
        unique[key] = match
    return list(unique.values())


async def fetch_tennis_com_matches(day: date) -> list[dict[str, str]]:
    from playwright.async_api import async_playwright

    url = TENNIS_COM_URL.format(date_key=day.isoformat())
    logging.info("Opening Tennis.com scores in headless Chromium: %s", url)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=USER_AGENT)
        response = await page.goto(url, wait_until="networkidle", timeout=60000)
        status = response.status if response else "unknown"
        logging.info("Tennis.com response status: %s", status)
        text = await page.locator("body").inner_text(timeout=15000)
        await browser.close()
    matches = parse_tennis_com_text(text, day)
    logging.info("Tennis.com singles matches parsed: %d", len(matches))
    return matches


async def fetch_matches(day: date) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}

    for source_name, fetcher in [("ESPN", fetch_espn_matches), ("Tennis.com", fetch_tennis_com_matches)]:
        try:
            source_matches = await fetcher(day)
        except Exception as exc:
            logging.warning("%s scrape failed: %s", source_name, exc)
            source_matches = []
        source_counts[source_name] = len(source_matches)
        for match in source_matches:
            if any(same_match(match, existing) for existing in matches):
                continue
            matches.append(match)

    logging.info(
        "Combined singles matches: %d (ESPN=%d, Tennis.com=%d)",
        len(matches),
        source_counts.get("ESPN", 0),
        source_counts.get("Tennis.com", 0),
    )
    return matches


def write_matches(matches: list[dict[str, str]], output: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install in .venv first.") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matches, columns=MATCH_COLUMNS).to_csv(output, index=False)
    logging.info("Saved %d real singles matches to %s", len(matches), output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ESPN tennis matches for today or tomorrow.")
    parser.add_argument("--target", choices=["today", "tomorrow"], required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


async def async_main() -> int:
    setup_logging()
    args = parse_args()
    try:
        day = target_date(args.target)
        matches = await fetch_matches(day)
        write_matches(matches, args.output)
        print(f"Browser opened headless: OK")
        print(f"Target date: {day.isoformat()}")
        print(f"Matches found: {len(matches)}")
        print(f"Saved CSV: {args.output}")
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        try:
            write_matches([], args.output)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))
