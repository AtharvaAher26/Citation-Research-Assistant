import httpx
import pytest

from citesage.generation.llm import (
    AllProvidersFailedError,
    GeminiClient,
    GroqClient,
    LLMConfigError,
    LLMRateLimitError,
    generate_with_fallback,
)


def _groq_body(text):
    return {"choices": [{"message": {"content": text}}]}


def _gemini_body(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_groq_success():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=_groq_body("grounded answer [1]"))

    client = GroqClient(api_key="test-key", transport=httpx.MockTransport(handler))
    assert client.generate("sys", "user") == "grounded answer [1]"


def test_gemini_success():
    def handler(request):
        assert request.headers["x-goog-api-key"] == "test-key"
        return httpx.Response(200, json=_gemini_body("gemini says [2]"))

    client = GeminiClient(api_key="test-key", transport=httpx.MockTransport(handler))
    assert client.generate("sys", "user") == "gemini says [2]"


def test_groq_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_groq_body("ok [1]"))

    client = GroqClient(
        api_key="k", transport=httpx.MockTransport(handler),
        max_attempts=3, backoff_base_seconds=0.0,
    )
    assert client.generate("s", "u") == "ok [1]"
    assert calls["n"] == 3


def test_groq_rate_limit_exhausts_to_error():
    client = GroqClient(
        api_key="k",
        transport=httpx.MockTransport(lambda request: httpx.Response(429, json={})),
        max_attempts=2, backoff_base_seconds=0.0,
    )
    with pytest.raises(LLMRateLimitError):
        client.generate("s", "u")


def test_bad_key_raises_config_error_without_retry():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={})

    client = GroqClient(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMConfigError):
        client.generate("s", "u")
    assert calls["n"] == 1  # no retries on auth failures


def test_missing_api_key_is_config_error():
    with pytest.raises(LLMConfigError):
        GroqClient(api_key="")


def test_generate_with_fallback_skips_broken_provider():
    class Boom:
        name = "boom"

        def generate(self, system, user):
            raise LLMRateLimitError("429 forever")

    good = GroqClient(
        api_key="k",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_groq_body("fallback ok [1]"))
        ),
    )
    text, name = generate_with_fallback([Boom(), good], "sys", "user")
    assert text == "fallback ok [1]"
    assert "groq" in name


def test_generate_with_fallback_raises_when_all_fail():
    class Boom:
        name = "boom"

        def generate(self, system, user):
            raise LLMRateLimitError("429 forever")

    with pytest.raises(AllProvidersFailedError):
        generate_with_fallback([Boom()], "sys", "user")