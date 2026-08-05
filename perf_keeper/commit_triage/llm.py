"""LLM client abstraction organized by spec, provider, and model."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel

from perf_keeper.commit_triage.config import (
    CONTEXT_BUDGET_FRACTION,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    EMBEDDING_MODEL,
    FALLBACK_CONTEXT_WINDOW,
    MAX_RETRIES,
)
from perf_keeper.commit_triage.models import (
    CommitRanking,
    FlashDecision,
    FlashResponse,
    FrontierResponse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CommitRanking",
    "FlashDecision",
    "FlashResponse",
    "FrontierResponse",
    "GEMINI_FLASH",
    "GEMINI_PRO",
    "GoogleLLMClient",
    "LLMClient",
    "ModelDef",
    "Provider",
    "Spec",
    "create_client",
]


class Spec(StrEnum):
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Provider(StrEnum):
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    VERTEX = "vertex"


@dataclass(frozen=True)
class ModelDef:
    model_id: str
    spec: Spec
    provider: Provider


GEMINI_FLASH = ModelDef("gemini-2.5-flash", Spec.GOOGLE, Provider.GOOGLE)
GEMINI_PRO = ModelDef("gemini-2.5-pro", Spec.GOOGLE, Provider.GOOGLE)

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Abstract LLM client with API-backed token counting, generation, and embedding."""

    def __init__(
        self,
        model: ModelDef,
        api_key: str,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self.context_window: int = 0

    def token_budget(self, fraction: float = CONTEXT_BUDGET_FRACTION) -> int:
        """Max tokens allowed per prompt at the given context fraction."""
        return int(self.context_window * fraction)

    @abstractmethod
    async def count_tokens(self, text: str) -> int: ...

    async def fits_in_budget(
        self, text: str, fraction: float = CONTEXT_BUDGET_FRACTION
    ) -> bool:
        return await self.count_tokens(text) <= self.token_budget(fraction)

    @abstractmethod
    async def complete(self, prompt: str) -> str: ...

    @abstractmethod
    async def complete_structured(self, prompt: str, schema: type[T]) -> T: ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class GoogleLLMClient(LLMClient):
    """Google GenAI client for Gemini models."""

    def __init__(
        self,
        model: ModelDef,
        api_key: str,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__(model, api_key, temperature)
        import google.genai as genai

        self._client = genai.Client(api_key=api_key)
        info = self._client.models.get(model=model.model_id)
        self.context_window = info.input_token_limit or FALLBACK_CONTEXT_WINDOW

    async def _request(self, fn):
        """Invoke fn() with retries on transient Google API errors."""
        for attempt in range(MAX_RETRIES):
            try:
                return await fn()
            except Exception as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if (
                    status
                    and (status == 429 or status >= 500)
                    and attempt < MAX_RETRIES - 1
                ):
                    logger.warning(
                        "Gemini API %d, retrying (attempt %d/%d)",
                        status,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    await asyncio.sleep(float(2**attempt))
                    continue
                raise
        raise RuntimeError("unreachable")

    async def count_tokens(self, text: str) -> int:
        response = await self._request(
            lambda: self._client.aio.models.count_tokens(
                model=self.model.model_id,
                contents=text,
            )
        )
        result = response.total_tokens or 0
        logger.debug("Token count for %s: %d tokens", self.model.model_id, result)
        return result

    async def complete(self, prompt: str) -> str:
        from google.genai import types

        logger.debug(
            "Generating with %s (temp=%.1f)", self.model.model_id, self._temperature
        )
        response = await self._request(
            lambda: self._client.aio.models.generate_content(
                model=self.model.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self._temperature,
                    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                ),
            )
        )
        return response.text or ""

    async def complete_structured(self, prompt: str, schema: type[T]) -> T:
        from google.genai import types

        logger.debug("Structured generation with %s", self.model.model_id)
        response = await self._request(
            lambda: self._client.aio.models.generate_content(
                model=self.model.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self._temperature,
                    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        )
        try:
            return schema.model_validate_json(response.text or "{}")
        except Exception:
            logger.warning("Structured response parse failed, using empty model")
            return schema.model_construct()

    async def embed(self, text: str) -> list[float]:
        logger.debug("Embedding with %s", EMBEDDING_MODEL)
        response = await self._request(
            lambda: self._client.aio.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
        )
        if not response.embeddings:
            return []
        return list(response.embeddings[0].values or [])


def create_client(
    model: ModelDef,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> LLMClient:
    """Instantiate the correct LLMClient for the given ModelDef."""
    if model.spec == Spec.GOOGLE:
        return GoogleLLMClient(model, api_key, temperature)
    raise NotImplementedError(f"Spec {model.spec!r} is not yet implemented")
