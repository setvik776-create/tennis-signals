import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predictor import normalize_surface, parse_dates  # noqa: E402


def test_parse_dates_handles_supported_mixed_formats():
    parsed = parse_dates(pd.Series(["20240601", "2024-06-02", "", "bad-date"]))

    assert parsed.iloc[0] == pd.Timestamp("2024-06-01")
    assert parsed.iloc[1] == pd.Timestamp("2024-06-02")
    assert pd.isna(parsed.iloc[2])
    assert pd.isna(parsed.iloc[3])


def test_normalize_surface_keeps_known_values_and_rejects_bad_values():
    assert normalize_surface("grass") == "Grass"
    assert normalize_surface("Hard") == "Hard"
    assert normalize_surface("Eva Asderaki-Moore") == "Unknown"
