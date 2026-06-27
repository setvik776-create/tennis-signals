#!/usr/bin/env python3
"""Small model router for the tennis enrichment agent."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelChoice:
    task: str
    model: str
    reason: str


class ModelRouter:
    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
        if base_url.endswith("/v1beta"):
            base_url = f"{base_url}/openai"
        self.base_url = base_url
        self.primary_model = os.getenv("GEMMA_PRIMARY_MODEL", "gemma-4-31b-it")
        self.secondary_model = os.getenv("GEMMA_SECONDARY_MODEL", "gemma-4-24b-a4b-it")
        self.light_model = os.getenv("GEMINI_LIGHT_MODEL", "gemini-3.1-flash-lite")
        self.embedding_model_1 = os.getenv("GEMINI_EMBEDDING_MODEL_1", "gemini-embedding-001")
        self.embedding_model_2 = os.getenv("GEMINI_EMBEDDING_MODEL_2", "gemini-embedding-002")

    def choose(self, task: str, ambiguity: float = 0.0) -> ModelChoice:
        if task in {"name_embedding", "player_similarity", "dedupe"}:
            return ModelChoice(task, self.embedding_model_1, "name matching and similarity search")
        if task in {"clean_scrape_text", "structure_text", "daily_probe"}:
            return ModelChoice(task, self.light_model, "cheap text cleanup and lightweight classification")
        if task in {"resolve_player_match", "uncertain_alias"} and ambiguity < 0.85:
            return ModelChoice(task, self.secondary_model, "medium ambiguity resolution")
        return ModelChoice(task, self.primary_model, "high ambiguity or audit decision")

    def chat(self, model: str, system: str, user: str, max_tokens: int = 120) -> dict:
        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_GEMINI_API_KEY/GEMINI_API_KEY/GOOGLE_API_KEY")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
                return {"status": response.status, "body": json.loads(body)}
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model API HTTP {exc.code}: {error_body[:500]}") from exc

    def daily_probe(self, candidate_count: int, staging_count: int) -> dict:
        choice = self.choose("daily_probe")
        result = self.chat(
            choice.model,
            "You are a strict data-quality checker. Reply with one short JSON object only.",
            (
                "Tennis enrichment daily probe. "
                f"candidate_count={candidate_count}, staging_count={staging_count}. "
                "Return JSON with keys status and note."
            ),
            max_tokens=80,
        )
        return {
            "task": choice.task,
            "model": choice.model,
            "reason": choice.reason,
            "status": result["status"],
            "raw": result["body"],
        }


def write_probe(path: Path, probe: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
