"""Optional AI explanations of the data quality results, using the Google Gemini API.

The app works fine without this. If GEMINI_API_KEY isn't set, the AI section
just shows setup instructions and everything else keeps working.
"""

import json
import os

# httpx is the HTTP library the Gemini SDK uses under the hood. It gets
# installed along with google-genai, and network failures (no internet,
# timeouts) come through as httpx errors rather than Gemini errors.
import httpx
from google import genai
from google.genai import errors, types

# A model that's available on the Gemini API free tier. If it hits a limit,
# we show a message instead of silently trying a different (possibly paid) model.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# The placeholder from .env.example. If someone copies the file without
# editing it, we treat the key as missing instead of sending a fake key.
PLACEHOLDER_API_KEY = "your_gemini_api_key_here"

# These instructions keep the model in its lane: explain the numbers we give
# it, don't invent new ones, don't pretend anything was fixed, and don't
# oversell what four basic checks can prove about the data.
AI_INSTRUCTIONS = """You are a data quality analyst explaining automated check results to a
non-technical business user.

Rules:
- Only use numbers that appear in the metrics provided. Never estimate or invent statistics.
- The data has NOT been cleaned or changed. Do not say or imply that it has been.
- Numeric outliers are possible anomalies, not confirmed errors.
- If a check found no issues, say so briefly instead of inventing problems.
- Only four checks were run: missing values, exact duplicate rows, leading/trailing
  whitespace, and IQR numeric outliers. Never describe the dataset as clean, error-free,
  accurate or ready to use, and never say it needs no cleaning. If nothing was found,
  say that these specific checks found no issues, and mention that things like invalid
  formats, impossible values, inconsistent spellings, near-duplicates and business-rule
  violations were not checked.

Respond in Markdown with these sections:
1. **Summary** - the main problems found, in plain language.
2. **Why it matters** - how these issues could affect reporting or analysis.
3. **Recommended next steps** - practical ways to clean or validate the data.
4. **Fix first** - which issues deserve attention first, and why.

Keep the whole response under 350 words."""


def get_api_key():
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_model_name():
    # Falls back to the default if GEMINI_MODEL is missing or blank
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def is_ai_configured():
    return get_api_key() not in ("", PLACEHOLDER_API_KEY)


def build_ai_payload(results):
    """Pick out only the aggregated metrics that are safe to send to Gemini.

    No cell values or row data are included - just counts, column names,
    data types and outlier thresholds. Row indexes are left out too since
    the model doesn't need them to explain the findings.
    """
    outlier_columns = {}
    for column, info in results["outliers"]["by_column"].items():
        outlier_columns[column] = {
            "outlier_count": info["outlier_count"],
            "lower_bound": info["lower_bound"],
            "upper_bound": info["upper_bound"],
        }

    return {
        "row_count": results["dataset_summary"]["row_count"],
        "column_count": results["dataset_summary"]["column_count"],
        "columns": results["dataset_summary"]["columns"],
        "missing_values": {
            "total_missing_cells": results["missing_values"]["total_missing_cells"],
            "rows_with_missing_values": results["missing_values"]["rows_with_missing_values"],
            "missing_cells_by_column": results["missing_values"]["by_column"],
        },
        "duplicate_row_count": results["duplicates"]["duplicate_row_count"],
        "whitespace_issues_by_column": results["whitespace"]["by_column"],
        "rows_with_numeric_outliers": results["outliers"]["rows_with_outliers"],
        "numeric_outliers_by_column": outlier_columns,
        "columns_skipped_for_outlier_check": results["outliers"]["skipped_columns"],
    }


def describe_gemini_error(error):
    """Turn a Gemini API error into a message a non-technical user can act on.

    Gemini errors carry an HTTP status code (error.code), which tells us
    what kind of problem it was.
    """
    message = error.message or ""

    # 429 = too many requests. On the free tier this means the per-minute or
    # per-day limit was reached. We deliberately don't retry with another model.
    if error.code == 429:
        return (
            "The Gemini free-tier limit has been reached. Please wait a minute and try again, "
            "or try again tomorrow if the daily limit was used up."
        )
    # Gemini reports a bad key as 400 "API key not valid"; 401/403 cover
    # keys that are blocked or lack permission.
    if error.code in (401, 403) or "API key" in message:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
    if error.code == 404:
        return (
            f"The model '{get_model_name()}' was not found or isn't available to your key. "
            "Check GEMINI_MODEL in your .env file."
        )
    if error.code >= 500:
        return "Gemini is temporarily unavailable. Please try again later."
    return f"Gemini returned an error: {message or error}"


def generate_ai_insights(results):
    """Send the aggregated metrics to Gemini and return the explanation as Markdown.

    Raises RuntimeError with a user-friendly message if anything goes wrong,
    so the app can show it without crashing.
    """
    if not is_ai_configured():
        raise RuntimeError("No Gemini API key is configured.")

    payload = build_ai_payload(results)
    prompt = (
        "Here are the results of automated data quality checks on a CSV dataset. "
        "Explain them following your instructions.\n\n"
        + json.dumps(payload, indent=2)
    )

    client = genai.Client(api_key=get_api_key())

    try:
        # generate_content sends one request and waits for the full answer.
        # system_instruction holds our rules; contents is the actual question.
        response = client.models.generate_content(
            model=get_model_name(),
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=AI_INSTRUCTIONS),
        )
    except errors.APIError as error:
        raise RuntimeError(describe_gemini_error(error))
    except httpx.RequestError:
        raise RuntimeError("Could not connect to Gemini. Check your internet connection.")

    # response.text can be None, e.g. if Gemini blocked the response for safety reasons
    if not response.text:
        raise RuntimeError("Gemini returned an empty response. Please try again.")

    return response.text
