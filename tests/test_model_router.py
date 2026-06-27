import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "enrichment_agent"))

from model_router import ModelRouter  # noqa: E402


def test_model_router_adds_openai_path_for_legacy_gemini_base_url(monkeypatch):
    monkeypatch.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

    router = ModelRouter()

    assert router.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
