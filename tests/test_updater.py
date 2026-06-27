import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from updater import infer_winner  # noqa: E402


def test_infer_winner_returns_none_when_scores_are_missing():
    text = """
    Final
    Player One
    Player Two
    """

    assert infer_winner(text) is None


def test_infer_winner_uses_parsed_set_scores():
    text = """
    Final
    Player One
    6
    4
    4
    Player Two
    3
    6
    6
    """

    assert infer_winner(text) == ("Player Two", "Player One")
