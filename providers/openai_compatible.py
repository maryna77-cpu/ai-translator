from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .base import BaseProvider, ProviderError, ProviderResult


def _error_details(exc: Exception) -> tuple[int | None, str | None, str]:
    status = getattr(exc, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None

    error_code = getattr(exc, "code", None)
    body: Any = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            error_code = error_code or err.get("code") or err.get("type")
            message = err.get("message") or str(exc)
        else:
            message = str(exc)
    else:
        message = str(exc)

    return status, str(error_code) if error_code else None, message


class OpenAICompatibleProvider(BaseProvider):
    """Provider for OpenAI-compatible gateways such as WinkAPI."""

    def __init__(self, cfg: dict):
        self.name = cfg.get("name", "openai_compatible")
        api_key = cfg.get("api_key")
        if not api_key and cfg.get("api_key_env"):
            api_key = os.getenv(cfg["api_key_env"])
        api_key = api_key or "not-needed"

        self.model = cfg["model"]
        self.base_url = cfg.get("base_url")
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=float(cfg.get("timeout_seconds", 180)),
            # Retry policy is controlled centrally by translator.py.
            max_retries=0,
        )
        self.max_tokens_field = cfg.get("max_tokens_field", "max_tokens")
        self.extra_body = cfg.get("extra_body", {})

    def generate(self, system_prompt, user_prompt, *, temperature, max_output_tokens):
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        kwargs[self.max_tokens_field] = max_output_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            status, error_code, message = _error_details(exc)
            raise ProviderError(
                message,
                status_code=status,
                error_code=error_code,
                raw=exc,
            ) from exc

        if not getattr(response, "choices", None):
            raise ProviderError("OpenAI-compatible API returned no choices.", raw=response)

        choice = response.choices[0]
        content = choice.message.content or ""
        if not content:
            raise ProviderError("OpenAI-compatible API returned an empty response.", raw=response)

        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }
        return ProviderResult(content=content, raw=response, usage=usage)
