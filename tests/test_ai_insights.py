"""Tests for the AI feature. The Gemini SDK is replaced with a fake, so no real API call is made."""

import json

import httpx
import pandas as pd
import pytest
from google.genai import errors

import ai_insights
import data_quality


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    """Stands in for client.models. Records each request, then returns or raises what we tell it to."""

    def __init__(self, result):
        self.result = result
        self.requests = []

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.result, Exception):
            raise self.result
        return FakeResponse(self.result)


def use_fake_gemini(monkeypatch, result):
    """Swap genai.Client for a fake one and return its FakeModels so tests can inspect requests."""
    fake_models = FakeModels(result)

    class FakeClient:
        def __init__(self, api_key):
            self.models = fake_models

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(ai_insights.genai, "Client", FakeClient)
    return fake_models


def make_gemini_error(code, message):
    # Same shape as the JSON error body the Gemini API sends back
    return errors.ClientError(code, {"error": {"code": code, "message": message}})


def make_messy_results():
    df = pd.DataFrame(
        {
            "customer_email": ["secret.person@example.com", " other@example.com", None, "x@example.com", "y@example.com"],
            "salary": [51000, 52000, 53000, 54000, 987654],
        }
    )
    return data_quality.run_all_checks(df)


# --- Configuration ---


def test_ai_is_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert ai_insights.is_ai_configured() is False
    with pytest.raises(RuntimeError):
        ai_insights.generate_ai_insights({})


def test_placeholder_key_from_env_example_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")

    assert ai_insights.is_ai_configured() is False


def test_default_model_is_used_when_not_configured(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert ai_insights.get_model_name() == "gemini-3.5-flash-lite"

    monkeypatch.setenv("GEMINI_MODEL", "some-other-model")
    assert ai_insights.get_model_name() == "some-other-model"


# --- Privacy ---


def test_payload_contains_no_cell_values():
    payload_text = json.dumps(ai_insights.build_ai_payload(make_messy_results()))

    # Column names are shared, but the actual values must not be
    assert "customer_email" in payload_text
    assert "secret.person" not in payload_text
    assert "other@example.com" not in payload_text
    assert "987654" not in payload_text


def test_request_sent_to_gemini_contains_no_cell_values(monkeypatch):
    fake_models = use_fake_gemini(monkeypatch, "Explanation")

    ai_insights.generate_ai_insights(make_messy_results())

    sent_prompt = fake_models.requests[0]["contents"]
    assert "customer_email" in sent_prompt
    assert "secret.person" not in sent_prompt
    assert "987654" not in sent_prompt


# --- Calling Gemini ---


def test_generate_ai_insights_returns_gemini_text(monkeypatch):
    fake_models = use_fake_gemini(monkeypatch, "## Summary\nThere are 2 missing cells.")

    text = ai_insights.generate_ai_insights(make_messy_results())

    assert text == "## Summary\nThere are 2 missing cells."
    assert fake_models.requests[0]["model"] == "gemini-3.5-flash-lite"
    assert fake_models.requests[0]["config"].system_instruction == ai_insights.AI_INSTRUCTIONS


def test_rate_limit_shows_friendly_message_and_does_not_switch_models(monkeypatch):
    fake_models = use_fake_gemini(monkeypatch, make_gemini_error(429, "Resource has been exhausted"))

    with pytest.raises(RuntimeError, match="free-tier limit"):
        ai_insights.generate_ai_insights(make_messy_results())

    # Exactly one request, to the configured model - no automatic fallback
    assert len(fake_models.requests) == 1
    assert fake_models.requests[0]["model"] == "gemini-3.5-flash-lite"


def test_invalid_api_key_shows_friendly_message(monkeypatch):
    use_fake_gemini(monkeypatch, make_gemini_error(400, "API key not valid. Please pass a valid API key."))

    with pytest.raises(RuntimeError, match="rejected the API key"):
        ai_insights.generate_ai_insights(make_messy_results())


def test_unknown_model_shows_friendly_message(monkeypatch):
    use_fake_gemini(monkeypatch, make_gemini_error(404, "models/made-up-model is not found"))

    with pytest.raises(RuntimeError, match="was not found"):
        ai_insights.generate_ai_insights(make_messy_results())


def test_network_failure_shows_friendly_message(monkeypatch):
    use_fake_gemini(monkeypatch, httpx.ConnectError("Connection refused"))

    with pytest.raises(RuntimeError, match="Could not connect"):
        ai_insights.generate_ai_insights(make_messy_results())


def test_empty_response_shows_friendly_message(monkeypatch):
    use_fake_gemini(monkeypatch, None)

    with pytest.raises(RuntimeError, match="empty response"):
        ai_insights.generate_ai_insights(make_messy_results())
