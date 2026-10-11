from __future__ import annotations

import json
import os
from string import Template

import requests

from .base import BaseProvider, ProviderError, ProviderResult


def _get_path(obj, path):
    current = obj
    for part in path:
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            raise KeyError(part)
    return current


class GenericJSONHTTPProvider(BaseProvider):
    """Generic adapter for JSON HTTP APIs that do not expose OpenAI-style chat."""

    def __init__(self, cfg: dict):
        self.name = cfg.get("name", "http_json")
        self.url = cfg["url"]
        self.timeout = float(cfg.get("timeout_seconds", 180))
        self.headers = dict(cfg.get("headers", {}))
        self.body_template = cfg.get("body_template", {
            "model": "${model}",
            "prompt": "${user_prompt}",
        })
        self.model = cfg.get("model", "")
        self.response_path = cfg.get("response_path", ["text"])

        for header_name, env_name in cfg.get("header_env", {}).items():
            value = os.getenv(env_name)
            if value:
                self.headers[header_name] = value

    def _render(self, obj, variables):
        if isinstance(obj, dict):
            return {k: self._render(v, variables) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._render(v, variables) for v in obj]
        if isinstance(obj, str):
            return Template(obj).safe_substitute(variables)
        return obj

    def generate(self, system_prompt, user_prompt, *, temperature, max_output_tokens):
        variables = {
            "model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": str(temperature),
            "max_output_tokens": str(max_output_tokens),
        }
        payload = self._render(self.body_template, variables)
        try:
            r = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            if not r.ok:
                raise ProviderError(
                    f"HTTP {r.status_code}: {r.text[:1000]}",
                    status_code=r.status_code,
                    raw=r.text,
                )
            data = r.json()
            content = _get_path(data, self.response_path)
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            return ProviderResult(content=content, raw=data)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(str(exc), raw=exc) from exc
