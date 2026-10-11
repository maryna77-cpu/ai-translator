from __future__ import annotations

import requests

from .base import BaseProvider, ProviderError, ProviderResult


class OllamaProvider(BaseProvider):
    def __init__(self, cfg: dict):
        self.name = cfg.get("name", "ollama")
        self.base_url = cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")
        self.model = cfg["model"]
        self.timeout = float(cfg.get("timeout_seconds", 600))

    def generate(self, system_prompt, user_prompt, *, temperature, max_output_tokens):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
        }
        try:
            r = requests.post(
                self.base_url + "/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            if not r.ok:
                raise ProviderError(
                    f"Ollama HTTP {r.status_code}: {r.text[:1000]}",
                    status_code=r.status_code,
                    raw=r.text,
                )
            data = r.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(str(exc), raw=exc) from exc

        content = data.get("message", {}).get("content", "")
        return ProviderResult(content=content, raw=data)
