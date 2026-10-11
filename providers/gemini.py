from __future__ import annotations

import os

from google import genai
from google.genai import types

from .base import BaseProvider, ProviderError, ProviderResult


class GeminiProvider(BaseProvider):
    """Native Google Gemini API adapter using the official google-genai SDK."""

    def __init__(self, cfg: dict):
        self.name = cfg.get("name", "gemini")
        api_key = cfg.get("api_key") or os.getenv(cfg.get("api_key_env", "GEMINI_API_KEY"))
        if not api_key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY or the configured api_key_env."
            )

        self.model = cfg["model"]
        self.client = genai.Client(api_key=api_key)
        self.use_schema = bool(cfg.get("structured_output", True))

    @staticmethod
    def _schema() -> dict:
        # Gemini accepts the JSON-schema equivalent directly.
        return {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "number": {"type": "INTEGER"},
                    "translation": {"type": "STRING"},
                },
                "required": ["number", "translation"],
                "propertyOrdering": ["number", "translation"],
            },
        }

    def generate(self, system_prompt, user_prompt, *, temperature, max_output_tokens):
        config_kwargs = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        # Gemini's native structured output prevents most malformed JSON responses.
        if self.use_schema:
            config_kwargs.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": self._schema(),
                }
            )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            raise ProviderError(str(exc), status_code=status, raw=exc) from exc

        content = getattr(response, "text", None) or ""
        if not content:
            raise ProviderError("Gemini returned an empty response.", raw=response)

        usage = None
        metadata = getattr(response, "usage_metadata", None)
        if metadata:
            usage = {
                "prompt_tokens": getattr(metadata, "prompt_token_count", None),
                "completion_tokens": getattr(metadata, "candidates_token_count", None),
                "total_tokens": getattr(metadata, "total_token_count", None),
            }

        return ProviderResult(content=content, raw=response, usage=usage)
