"""Tests for LLM client abstraction."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from perf_keeper.commit_triage.llm import (
    GEMINI_FLASH,
    GEMINI_PRO,
    GoogleLLMClient,
    ModelDef,
    Provider,
    Spec,
    create_client,
)
from perf_keeper.commit_triage.models import (
    FlashResponse,
)


# ModelDef and constants


class TestModelDef:
    def test_gemini_flash_attributes(self):
        assert GEMINI_FLASH.model_id == "gemini-2.5-flash"
        assert GEMINI_FLASH.spec == Spec.GOOGLE
        assert GEMINI_FLASH.provider == Provider.GOOGLE

    def test_gemini_pro_attributes(self):
        assert GEMINI_PRO.model_id == "gemini-2.5-pro"
        assert GEMINI_PRO.spec == Spec.GOOGLE

    def test_frozen_dataclass(self):
        with pytest.raises(Exception):
            setattr(GEMINI_FLASH, "model_id", "other")

    def test_custom_model_def(self):
        m = ModelDef("my-model", Spec.OPENAI, Provider.OPENAI)
        assert m.model_id == "my-model"
        assert m.spec == Spec.OPENAI


class TestSpecAndProvider:
    def test_spec_values(self):
        assert Spec.GOOGLE == "google"
        assert Spec.OPENAI == "openai"
        assert Spec.ANTHROPIC == "anthropic"

    def test_provider_values(self):
        assert Provider.GOOGLE == "google"
        assert Provider.VERTEX == "vertex"
        assert Provider.AZURE == "azure"


# LLMClient base


class TestLLMClientBase:
    def _client(self, context_window=1_000_000):
        from perf_keeper.commit_triage.tests.conftest import MockLLMClient

        return MockLLMClient(context_window=context_window)

    def test_token_budget_sixty_percent(self):
        assert self._client(1_000_000).token_budget(0.6) == 600_000

    def test_token_budget_custom_fraction(self):
        assert self._client(100_000).token_budget(0.5) == 50_000

    async def test_fits_in_budget_true(self):
        assert await self._client(10_000).fits_in_budget("short text", fraction=0.6)

    async def test_fits_in_budget_false(self):
        assert not await self._client(100).fits_in_budget("x" * 10_000, fraction=0.6)

    async def test_count_tokens_char_based(self):
        count = await self._client().count_tokens("x" * 400)
        assert count == 100


# GoogleLLMClient


def _setup_mock_genai(input_token_limit=1_000_000):
    """Patch google.genai.Client and return (mock_genai_client, patcher)."""
    mock_gc = MagicMock()
    mock_info = MagicMock()
    mock_info.input_token_limit = input_token_limit
    mock_gc.models.get.return_value = mock_info
    return mock_gc


class TestGoogleLLMClient:
    def test_context_window_fetched_from_api(self):
        mock_gc = _setup_mock_genai(500_000)
        with patch("google.genai.Client", return_value=mock_gc):
            client = GoogleLLMClient(GEMINI_FLASH, "test-key")
        assert client.context_window == 500_000
        mock_gc.models.get.assert_called_once_with(model=GEMINI_FLASH.model_id)

    def test_context_window_none_falls_back(self):
        mock_gc = _setup_mock_genai(None)
        with patch("google.genai.Client", return_value=mock_gc):
            client = GoogleLLMClient(GEMINI_FLASH, "test-key")
        assert client.context_window == 1_000_000

    async def test_count_tokens_calls_api(self):
        mock_gc = _setup_mock_genai()
        mock_count_resp = MagicMock()
        mock_count_resp.total_tokens = 42
        mock_gc.aio.models.count_tokens = AsyncMock(return_value=mock_count_resp)

        with patch("google.genai.Client", return_value=mock_gc):
            client = GoogleLLMClient(GEMINI_FLASH, "test-key")

        count = await client.count_tokens("Hello world")
        assert count == 42
        mock_gc.aio.models.count_tokens.assert_called_once()

    async def test_count_tokens_none_returns_zero(self):
        mock_gc = _setup_mock_genai()
        mock_count_resp = MagicMock()
        mock_count_resp.total_tokens = None
        mock_gc.aio.models.count_tokens = AsyncMock(return_value=mock_count_resp)

        with patch("google.genai.Client", return_value=mock_gc):
            client = GoogleLLMClient(GEMINI_FLASH, "test-key")

        assert await client.count_tokens("text") == 0

    async def test_complete_returns_text(self):
        mock_gc = _setup_mock_genai()
        mock_resp = MagicMock()
        mock_resp.text = "Generated text"
        mock_gc.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        with patch("google.genai.Client", return_value=mock_gc):
            with patch("google.genai.types"):
                client = GoogleLLMClient(GEMINI_FLASH, "test-key")
                text = await client.complete("prompt")

        assert text == "Generated text"

    async def test_complete_none_text_returns_empty(self):
        mock_gc = _setup_mock_genai()
        mock_resp = MagicMock()
        mock_resp.text = None
        mock_gc.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        with patch("google.genai.Client", return_value=mock_gc):
            with patch("google.genai.types"):
                client = GoogleLLMClient(GEMINI_FLASH, "test-key")
                text = await client.complete("prompt")

        assert text == ""

    async def test_complete_structured_parses_json(self):
        mock_gc = _setup_mock_genai()
        json_text = (
            '{"decisions": [{"commit_key": "a:b", "worth_investigating": true}]}'
        )
        mock_resp = MagicMock()
        mock_resp.text = json_text
        mock_gc.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        with patch("google.genai.Client", return_value=mock_gc):
            with patch("google.genai.types"):
                client = GoogleLLMClient(GEMINI_FLASH, "test-key")
                result = await client.complete_structured("prompt", FlashResponse)

        assert isinstance(result, FlashResponse)
        assert len(result.decisions) == 1
        assert result.decisions[0].commit_key == "a:b"
        assert result.decisions[0].worth_investigating is True

    async def test_embed_returns_float_list(self):
        mock_gc = _setup_mock_genai()
        mock_emb_resp = MagicMock()
        mock_emb_resp.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
        mock_gc.aio.models.embed_content = AsyncMock(return_value=mock_emb_resp)

        with patch("google.genai.Client", return_value=mock_gc):
            client = GoogleLLMClient(GEMINI_FLASH, "test-key")

        result = await client.embed("some text")
        assert isinstance(result, list)
        assert result == pytest.approx([0.1, 0.2, 0.3])

    async def test_embed_uses_first_embedding_in_list(self):
        mock_gc = _setup_mock_genai()
        mock_emb_resp = MagicMock()
        mock_emb_resp.embeddings = [
            MagicMock(values=[0.9, 0.1]),
            MagicMock(values=[0.1, 0.9]),
        ]
        mock_gc.aio.models.embed_content = AsyncMock(return_value=mock_emb_resp)

        with patch("google.genai.Client", return_value=mock_gc):
            client = GoogleLLMClient(GEMINI_FLASH, "test-key")

        result = await client.embed("text")
        assert result == pytest.approx([0.9, 0.1])


# create_client factory


class TestCreateClient:
    def test_google_spec_returns_google_client(self):
        mock_gc = _setup_mock_genai()
        with patch("google.genai.Client", return_value=mock_gc):
            client = create_client(GEMINI_FLASH, "test-key")
        assert isinstance(client, GoogleLLMClient)

    def test_unsupported_spec_raises_not_implemented(self):
        model = ModelDef("gpt-4o", Spec.OPENAI, Provider.OPENAI)
        with pytest.raises(NotImplementedError, match="openai"):
            create_client(model, "key")
