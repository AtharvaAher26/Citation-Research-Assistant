"""LLM providers behind one interface: Groq (primary) + Gemini (fallback).

Both are called over plain HTTPS with httpx (no vendor SDKs), with strict
timeouts, bounded retries on transient failures, and typed errors so the
fallback chain and the CLI can react precisely.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, Sequence

import httpx

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_DEFAULT_ATTEMPTS = 3


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMConfigError(LLMError):
    """Missing/invalid credentials or misconfiguration - never retry."""


class LLMRateLimitError(LLMError):
    """Provider kept returning 429 after all retries."""


class AllProvidersFailedError(LLMError):
    """Every configured provider failed."""


class LLMClient(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str: ...


def _extract_groq_text(body: dict[str, Any]) -> str:
    try:
        return str(body["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_gemini_text(body: dict[str, Any]) -> str:
    try:
        parts = body["candidates"][0]["content"]["parts"]
        return "".join(str(part.get("text", "")) for part in parts)
    except (KeyError, IndexError, TypeError):
        return ""


class _BaseClient:
    """Shared retry/timeout plumbing for both providers."""

    name: str

    def __init__(
        self,
        timeout_seconds: float,
        max_output_tokens: int,
        transport: httpx.BaseTransport | None,
        max_attempts: int,
        backoff_base_seconds: float,
    ) -> None:
        self._timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=10.0, pool=10.0)
        self._max_output_tokens = max_output_tokens
        self._transport = transport
        self._max_attempts = max(1, max_attempts)
        self._backoff_base = max(0.0, backoff_base_seconds)

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        extractor: Any,
    ) -> str:
        last_error: LLMError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                    response = client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = LLMError(f"{self.name}: network error: {exc}")
                logger.warning("%s (attempt %d/%d)", last_error, attempt, self._max_attempts)
            else:
                if response.status_code == 200:
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise LLMError(f"{self.name}: invalid JSON in response: {exc}") from exc
                    text = extractor(body)
                    if not text:
                        raise LLMError(f"{self.name}: response contained no text")
                    return text
                if response.status_code in (401, 403):
                    raise LLMConfigError(
                        f"{self.name}: auth failed ({response.status_code}) - check your API key"
                    )
                if response.status_code == 400:
                    raise LLMError(f"{self.name}: bad request - {response.text[:300]}")
                if response.status_code in _RETRYABLE_STATUS:
                    if response.status_code == 429:
                        last_error = LLMRateLimitError(f"{self.name}: rate limited (429)")
                    else:
                        last_error = LLMError(f"{self.name}: server error ({response.status_code})")
                    logger.warning("%s (attempt %d/%d)", last_error, attempt, self._max_attempts)
                else:
                    raise LLMError(
                        f"{self.name}: unexpected status {response.status_code}: {response.text[:300]}"
                    )
            if attempt < self._max_attempts:
                time.sleep(self._backoff_base * attempt)
        raise last_error or LLMError(f"{self.name}: failed after {self._max_attempts} attempts")


class GroqClient(_BaseClient):
    """OpenAI-compatible chat completions on Groq's free tier."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 1024,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = _DEFAULT_ATTEMPTS,
        backoff_base_seconds: float = 2.0,
    ) -> None:
        if not api_key:
            raise LLMConfigError("Groq API key is missing (set CITESAGE_GROQ_API_KEY in .env)")
        self.name = f"groq/{model}"
        self.model = model
        super().__init__(timeout_seconds, max_output_tokens, transport, max_attempts, backoff_base_seconds)
        self._api_key = api_key

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": self._max_output_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        return self._post_json(GROQ_CHAT_URL, payload, headers, _extract_groq_text)


class GeminiClient(_BaseClient):
    """Gemini generateContent API on Google AI Studio's free tier."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 1024,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = _DEFAULT_ATTEMPTS,
        backoff_base_seconds: float = 2.0,
    ) -> None:
        if not api_key:
            raise LLMConfigError("Gemini API key is missing (set CITESAGE_GEMINI_API_KEY in .env)")
        self.name = f"gemini/{model}"
        self.model = model
        super().__init__(timeout_seconds, max_output_tokens, transport, max_attempts, backoff_base_seconds)
        self._api_key = api_key

    def generate(self, system: str, user: str) -> str:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": self._max_output_tokens},
        }
        url = GEMINI_GENERATE_URL.format(model=self.model)
        headers = {"x-goog-api-key": self._api_key}
        return self._post_json(url, payload, headers, _extract_gemini_text)


def build_clients(
    groq_api_key: str = "",
    gemini_api_key: str = "",
    groq_model: str = DEFAULT_GROQ_MODEL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    provider: str = "auto",
    timeout_seconds: float = 30.0,
    max_output_tokens: int = 1024,
) -> list[LLMClient]:
    """Ordered provider chain per user preference and key availability."""
    if provider == "groq":
        return [
            GroqClient(
                api_key=groq_api_key, model=groq_model,
                timeout_seconds=timeout_seconds, max_output_tokens=max_output_tokens,
            )
        ]
    if provider == "gemini":
        return [
            GeminiClient(
                api_key=gemini_api_key, model=gemini_model,
                timeout_seconds=timeout_seconds, max_output_tokens=max_output_tokens,
            )
        ]
    if provider != "auto":
        raise LLMConfigError(f"unknown provider {provider!r} (use auto | groq | gemini)")

    clients: list[LLMClient] = []
    if groq_api_key:
        clients.append(
            GroqClient(
                api_key=groq_api_key, model=groq_model,
                timeout_seconds=timeout_seconds, max_output_tokens=max_output_tokens,
            )
        )
    if gemini_api_key:
        clients.append(
            GeminiClient(
                api_key=gemini_api_key, model=gemini_model,
                timeout_seconds=timeout_seconds, max_output_tokens=max_output_tokens,
            )
        )
    if not clients:
        raise LLMConfigError(
            "No LLM API keys configured. Add CITESAGE_GROQ_API_KEY (https://console.groq.com/keys) "
            "and/or CITESAGE_GEMINI_API_KEY (https://aistudio.google.com/apikey) to your .env"
        )
    return clients


def generate_with_fallback(
    clients: Sequence[LLMClient], system: str, user: str
) -> tuple[str, str]:
    """Try each client in order; return (text, client_name) or raise AllProvidersFailedError."""
    errors: list[str] = []
    for client in clients:
        try:
            return client.generate(system, user), client.name
        except LLMConfigError as exc:
            errors.append(str(exc))
            logger.warning("Skipping provider: %s", exc)
        except LLMError as exc:
            errors.append(str(exc))
            logger.warning("Provider failed, trying next: %s", exc)
    raise AllProvidersFailedError("All providers failed: " + " | ".join(errors))