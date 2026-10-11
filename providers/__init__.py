from __future__ import annotations

from .base import BaseProvider, ProviderError, ProviderResult


def build_provider(cfg: dict, provider_name: str) -> BaseProvider:
    if provider_name not in cfg["providers"]:
        raise ValueError(f"Provider '{provider_name}' is not defined in config.json")

    item = dict(cfg["providers"][provider_name])
    item["name"] = provider_name
    kind = item.get("type", "openai_compatible")

    # Import only the selected adapter so unused providers do not require their SDKs.
    if kind == "openai_compatible":
        from .openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(item)
    if kind == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(item)
    if kind == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(item)
    if kind == "http_json":
        from .http_json import GenericJSONHTTPProvider
        return GenericJSONHTTPProvider(item)
    raise ValueError(f"Unsupported provider type: {kind}")


__all__ = ["BaseProvider", "ProviderError", "ProviderResult", "build_provider"]
