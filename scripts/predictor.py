#!/usr/bin/env python3
"""Predict tomorrow's tennis matches using H2H, surface win rate, and recent form."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable


BASE_DIR = Path("/root/tennis_signals")
DEFAULT_HISTORY = BASE_DIR / "data" / "tennis_all_matches_2024_to_now.csv"
DEFAULT_MATCHES = BASE_DIR / "data" / "tomorrow_matches.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "predictions.csv"
DEFAULT_MODEL = BASE_DIR / "data" / "tennis_model.joblib"
LOG_FILE = BASE_DIR / "logs" / "predictor.log"
FEATURES = ["h2h", "p1_surface_wr", "p2_surface_wr", "p1_form", "p2_form", "p1_fatigue", "p2_fatigue"]
pd = None


def require_pandas():
    global pd
    if pd is None:
        try:
            import pandas as pandas_module
        except ImportError as exc:
            raise RuntimeError("pandas is required. Install with: python3 -m pip install pandas") from exc
        pd = pandas_module
    return pd


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )


def first_existing(columns: Iterable[str], options: list[str]) -> str | None:
    lower = {column.lower().strip(): column for column in columns}
    for option in options:
        if option in lower:
            return lower[option]
    return None


def parse_dates(values: pd.Series) -> pd.Series:
    raw = values.fillna("").astype(str).str.strip()
    ymd = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    generic = pd.to_datetime(raw, errors="coerce")
    return ymd.fillna(generic)


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "match_date": ["match_date", "date", "tourney_date", "start_date"],
        "tournament": ["tournament", "tourney_name", "event", "competition"],
        "surface": ["surface", "court_surface", "court"],
        "player1": ["player1", "player_1", "p1", "winner", "home", "player_a"],
        "player2": ["player2", "player_2", "p2", "loser", "away", "player_b"],
        "winner": ["winner", "match_winner", "winning_player"],
    }
    mapped: dict[str, pd.Series] = {}
    for target, names in aliases.items():
        source = first_existing(df.columns, names)
        if source:
            mapped[target] = df[source]

    if "winner" not in mapped and {"player1", "player2"}.issubset(mapped):
        raise ValueError("History CSV must contain a winner column or winner/loser style columns.")
    if "player1" not in mapped and first_existing(df.columns, ["winner"]):
        mapped["player1"] = df[first_existing(df.columns, ["winner"])]
    if "player2" not in mapped and first_existing(df.columns, ["loser"]):
        mapped["player2"] = df[first_existing(df.columns, ["loser"])]

    required = ["player1", "player2", "winner"]
    missing = [field for field in required if field not in mapped]
    if missing:
        raise ValueError(f"History CSV missing required fields after normalization: {missing}")

    out = pd.DataFrame(mapped)
    out["match_date"] = parse_dates(out.get("match_date"))
    out["surface"] = out.get("surface", "").fillna("").astype(str)
    out["tournament"] = out.get("tournament", "").fillna("").astype(str)
    for col in ["player1", "player2", "winner"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out = out[(out["player1"] != "") & (out["player2"] != "") & (out["winner"] != "") & (out["player1"] != out["player2"])]
    return out.sort_values("match_date", na_position="first").reset_index(drop=True)


def normalize_matches(df: pd.DataFrame) -> pd.DataFrame:
    required = ["match_date", "tournament", "surface", "player1", "player2"]
    lower = {column.lower().strip(): column for column in df.columns}
    missing = [field for field in required if field not in lower]
    if missing:
        raise ValueError(f"Tomorrow matches CSV missing columns: {missing}")
    out = pd.DataFrame({field: df[lower[field]] for field in required})
    for field in required:
        out[field] = out[field].fillna("").astype(str).str.strip()
    out["match_date"] = parse_dates(out["match_date"])
    return out[(out["player1"] != "") & (out["player2"] != "")]


def add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p1 = out["player1"].astype(str)
    p2 = out["player2"].astype(str)
    out["surface_key"] = out["surface"].fillna("").astype(str).str.lower()
    out["pair_key"] = p1.where(p1 <= p2, p2) + "||" + p2.where(p1 <= p2, p1)
    return out


def player_long(history: pd.DataFrame) -> pd.DataFrame:
    base = history[["row_id", "match_date", "surface_key"]]
    p1 = base.assign(player=history["player1"], win=(history["winner"] == history["player1"]).astype(float))
    p2 = base.assign(player=history["player2"], win=(history["winner"] == history["player2"]).astype(float))
    long = pd.concat([p1, p2], ignore_index=True).sort_values(["player", "match_date", "row_id"])
    long["prior_win"] = long.groupby("player")["win"].shift()
    rolling = long.groupby("player")["prior_win"].rolling(5, min_periods=1)
    long["form"] = (rolling.sum() / rolling.count()).reset_index(level=0, drop=True).fillna(0.5)
    long["surface_played"] = long.groupby(["player", "surface_key"]).cumcount()
    long["surface_wins"] = long.groupby(["player", "surface_key"])["win"].cumsum() - long["win"]
    long["surface_wr"] = (long["surface_wins"] / long["surface_played"]).fillna(0.5)
    prev_date = long.groupby("player")["match_date"].shift()
    rest_days = (long["match_date"] - prev_date).dt.days.clip(lower=0)
    long["fatigue"] = (1.0 / (rest_days + 1.0)).fillna(0.0)
    return long


def h2h_training_features(history: pd.DataFrame) -> pd.DataFrame:
    p1 = history[["row_id", "pair_key"]].assign(player=history["player1"], win=(history["winner"] == history["player1"]).astype(float))
    p2 = history[["row_id", "pair_key"]].assign(player=history["player2"], win=(history["winner"] == history["player2"]).astype(float))
    pair_long = pd.concat([p1, p2], ignore_index=True).sort_values(["pair_key", "row_id"])
    pair_long["prior_wins"] = pair_long.groupby(["pair_key", "player"])["win"].cumsum() - pair_long["win"]
    p1_wins = pair_long.rename(columns={"player": "player1", "prior_wins": "p1_h2h_wins"})[
        ["row_id", "pair_key", "player1", "p1_h2h_wins"]
    ]
    p2_wins = pair_long.rename(columns={"player": "player2", "prior_wins": "p2_h2h_wins"})[
        ["row_id", "pair_key", "player2", "p2_h2h_wins"]
    ]
    h2h = history[["row_id", "pair_key", "player1", "player2"]].merge(p1_wins, how="left")
    h2h = h2h.merge(p2_wins, how="left")
    games = history[["row_id"]].assign(pair_games=history.groupby("pair_key").cumcount())
    h2h = h2h.merge(games, on="row_id", how="left")
    h2h["h2h"] = ((h2h["p1_h2h_wins"] - h2h["p2_h2h_wins"]) / h2h["pair_games"]).replace(
        [float("inf"), float("-inf")], 0.0
    ).fillna(0.0)
    return h2h[["row_id", "h2h"]]


def build_training_features(history: pd.DataFrame) -> pd.DataFrame:
    prepared = add_match_keys(history).reset_index(drop=True)
    prepared["row_id"] = prepared.index
    valid = prepared[prepared["winner"].eq(prepared["player1"]) | prepared["winner"].eq(prepared["player2"])].copy()
    long = player_long(valid)
    p1_features = long.rename(
        columns={"player": "player1", "surface_wr": "p1_surface_wr", "form": "p1_form", "fatigue": "p1_fatigue"}
    )[["row_id", "player1", "p1_surface_wr", "p1_form", "p1_fatigue"]]
    p2_features = long.rename(
        columns={"player": "player2", "surface_wr": "p2_surface_wr", "form": "p2_form", "fatigue": "p2_fatigue"}
    )[["row_id", "player2", "p2_surface_wr", "p2_form", "p2_fatigue"]]
    features = valid[["row_id", "player1", "player2", "winner"]].merge(p1_features, how="left")
    features = features.merge(p2_features, how="left")
    features = features.merge(h2h_training_features(valid), on="row_id", how="left")
    features["label"] = (features["winner"] == features["player1"]).astype(int)
    features = features.fillna(
        {
            "h2h": 0.0,
            "p1_surface_wr": 0.5,
            "p2_surface_wr": 0.5,
            "p1_form": 0.5,
            "p2_form": 0.5,
            "p1_fatigue": 0.0,
            "p2_fatigue": 0.0,
        }
    )
    reversed_features = features.copy()
    reversed_features["h2h"] = -features["h2h"]
    reversed_features["p1_surface_wr"] = features["p2_surface_wr"]
    reversed_features["p2_surface_wr"] = features["p1_surface_wr"]
    reversed_features["p1_form"] = features["p2_form"]
    reversed_features["p2_form"] = features["p1_form"]
    reversed_features["p1_fatigue"] = features["p2_fatigue"]
    reversed_features["p2_fatigue"] = features["p1_fatigue"]
    reversed_features["label"] = 1 - features["label"]
    return pd.concat([features, reversed_features], ignore_index=True)


def latest_player_features(history: pd.DataFrame) -> pd.DataFrame:
    prepared = add_match_keys(history).reset_index(drop=True)
    prepared["row_id"] = prepared.index
    long = player_long(prepared)
    latest = long.sort_values(["player", "match_date", "row_id"]).drop_duplicates("player", keep="last")
    return latest[["player", "form", "match_date"]].rename(columns={"form": "latest_form", "match_date": "last_match_date"})


def surface_history_features(history: pd.DataFrame) -> pd.DataFrame:
    prepared = add_match_keys(history).reset_index(drop=True)
    prepared["row_id"] = prepared.index
    long = player_long(prepared)
    totals = long.groupby(["player", "surface_key"], as_index=False).agg(surface_wins=("win", "sum"), surface_played=("win", "count"))
    totals["surface_wr"] = (totals["surface_wins"] / totals["surface_played"]).fillna(0.5)
    return totals[["player", "surface_key", "surface_wr"]]


def h2h_history_features(history: pd.DataFrame) -> pd.DataFrame:
    prepared = add_match_keys(history).reset_index(drop=True)
    prepared["row_id"] = prepared.index
    p1 = prepared[["pair_key"]].assign(player=prepared["player1"], win=(prepared["winner"] == prepared["player1"]).astype(float))
    p2 = prepared[["pair_key"]].assign(player=prepared["player2"], win=(prepared["winner"] == prepared["player2"]).astype(float))
    totals = pd.concat([p1, p2], ignore_index=True).groupby(["pair_key", "player"], as_index=False)["win"].sum()
    return totals.rename(columns={"win": "h2h_wins"})


def build_prediction_features(history: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    prepared = add_match_keys(matches).reset_index(drop=True)
    prepared["match_id"] = prepared.index

    latest = latest_player_features(history)
    surface = surface_history_features(history)
    h2h = h2h_history_features(history)

    p1_latest = latest.rename(columns={"player": "player1", "latest_form": "p1_form", "last_match_date": "p1_last_match_date"})
    p2_latest = latest.rename(columns={"player": "player2", "latest_form": "p2_form", "last_match_date": "p2_last_match_date"})
    p1_surface = surface.rename(columns={"player": "player1", "surface_wr": "p1_surface_wr"})
    p2_surface = surface.rename(columns={"player": "player2", "surface_wr": "p2_surface_wr"})
    p1_h2h = h2h.rename(columns={"player": "player1", "h2h_wins": "p1_h2h_wins"})
    p2_h2h = h2h.rename(columns={"player": "player2", "h2h_wins": "p2_h2h_wins"})

    features = prepared.merge(p1_latest, on="player1", how="left").merge(p2_latest, on="player2", how="left")
    features = features.merge(p1_surface, on=["player1", "surface_key"], how="left")
    features = features.merge(p2_surface, on=["player2", "surface_key"], how="left")
    features = features.merge(p1_h2h, on=["pair_key", "player1"], how="left")
    features = features.merge(p2_h2h, on=["pair_key", "player2"], how="left")

    h2h_total = features["p1_h2h_wins"].fillna(0.0) + features["p2_h2h_wins"].fillna(0.0)
    features["h2h"] = (
        (features["p1_h2h_wins"].fillna(0.0) - features["p2_h2h_wins"].fillna(0.0)) / h2h_total
    ).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    p1_rest = (features["match_date"] - features["p1_last_match_date"]).dt.days.clip(lower=0)
    p2_rest = (features["match_date"] - features["p2_last_match_date"]).dt.days.clip(lower=0)
    features["p1_fatigue"] = (1.0 / (p1_rest + 1.0)).fillna(0.0)
    features["p2_fatigue"] = (1.0 / (p2_rest + 1.0)).fillna(0.0)

    return features.fillna(
        {
            "h2h": 0.0,
            "p1_surface_wr": 0.5,
            "p2_surface_wr": 0.5,
            "p1_form": 0.5,
            "p2_form": 0.5,
            "p1_fatigue": 0.0,
            "p2_fatigue": 0.0,
        }
    )


def heuristic_probability(features: pd.DataFrame) -> pd.Series:
    score = (
        0.23 * (features["h2h"] + 1.0) / 2.0
        + 0.22 * features["p1_surface_wr"]
        + 0.22 * features["p1_form"]
        + 0.11 * (1.0 - features["p2_surface_wr"])
        + 0.11 * (1.0 - features["p2_form"])
        + 0.055 * (1.0 - features["p1_fatigue"])
        + 0.055 * features["p2_fatigue"]
    )
    return score.clip(lower=0.05, upper=0.95)


def train_model(history: pd.DataFrame, model_out: Path):
    try:
        from joblib import dump
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        logging.warning("sklearn/joblib not available, using heuristic fallback: %s", exc)
        return None

    training = build_training_features(history)
    if len(training) < 50 or training["label"].nunique() < 2:
        logging.warning("Not enough labeled data for ML (%s rows), using heuristic fallback.", len(training))
        return None

    x = training[FEATURES]
    y = training["label"]
    stratify = y if y.value_counts().min() >= 2 else None
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=3)),
        ]
    )
    model.fit(train_x, train_y)
    logging.info("ML model trained. Holdout accuracy: %.3f", model.score(test_x, test_y))
    model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_out)
    return model


def predict(history: pd.DataFrame, matches: pd.DataFrame, model) -> pd.DataFrame:
    features = build_prediction_features(history, matches)
    if model is not None:
        p1_prob = pd.Series(model.predict_proba(features[FEATURES])[:, 1], index=features.index)
        mode = "random_forest"
    else:
        p1_prob = heuristic_probability(features)
        mode = "heuristic"
    p2_prob = 1.0 - p1_prob
    output = features[["match_date", "tournament", "surface", "player1", "player2"]].copy()
    output["predicted_winner"] = features["player1"].where(p1_prob >= p2_prob, features["player2"])
    output["player1_win_probability"] = p1_prob.round(4)
    output["player2_win_probability"] = p2_prob.round(4)
    output["confidence"] = pd.concat([p1_prob, p2_prob], axis=1).max(axis=1).round(4)
    output["model_mode"] = mode
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tennis match winners.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    try:
        require_pandas()
        if not args.history.exists():
            raise FileNotFoundError(f"History CSV not found: {args.history}")
        if not args.matches.exists():
            raise FileNotFoundError(f"Tomorrow matches CSV not found: {args.matches}")
        history = normalize_history(pd.read_csv(args.history))
        matches = normalize_matches(pd.read_csv(args.matches))
        if matches.empty:
            raise ValueError("Tomorrow matches CSV has no valid rows.")
        model = train_model(history, args.model_out)
        predictions = predict(history, matches, model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        logging.info("Wrote %s predictions to %s", len(predictions), args.output)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
