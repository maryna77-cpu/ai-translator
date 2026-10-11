from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderResult:
    content: str
    raw: Any = None
    usage: dict[str, Any] | None = None


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        raw: Any = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.raw = raw
        self.error_code = error_code


class BaseProvider(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> ProviderResult:
        raise NotImplementedError
