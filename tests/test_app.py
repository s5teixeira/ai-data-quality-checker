"""Smoke tests that run the Streamlit app headlessly with Streamlit's AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import ai_insights

# AppTest resolves relative paths from this test file, so point one folder up
APP_PATH = "../app.py"
SAMPLE_CSV = Path(__file__).parent.parent / "sample_data" / "sample_sales.csv"


def start_app(monkeypatch):
    # An empty key means "not configured". load_dotenv() won't overwrite a
    # variable that's already set, so a local .env can't affect this test.
    monkeypatch.setenv("GEMINI_API_KEY", "")

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    return app


def click_button(app, label):
    button = next(button for button in app.button if button.label == label)
    button.click().run()


def button_labels(app):
    return [button.label for button in app.button]


def run_app_with_sample_data(monkeypatch):
    app = start_app(monkeypatch)
    click_button(app, "Load Sample Dataset")
    return app


def test_app_starts_with_only_the_two_dataset_buttons(monkeypatch):
    app = start_app(monkeypatch)

    assert not app.exception
    assert button_labels(app) == ["Load Sample Dataset"]
    assert len(app.file_uploader) == 1  # inside the "Upload CSV" popover
    assert len(app.metric) == 0


def test_sample_dataset_is_analyzed(monkeypatch):
    app = run_app_with_sample_data(monkeypatch)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Rows"] == "44"
    assert metrics["Duplicate rows"] == "2"
    assert [tab.label for tab in app.tabs] == ["Overview", "Detailed Findings", "Explore Data", "AI Insights"]


def test_loaded_dataset_hides_upload_buttons_and_shows_delete(monkeypatch):
    app = run_app_with_sample_data(monkeypatch)

    assert "Delete Dataset" in button_labels(app)
    assert "Load Sample Dataset" not in button_labels(app)
    assert len(app.file_uploader) == 0


def test_delete_dataset_returns_to_start_and_allows_reloading(monkeypatch):
    app = run_app_with_sample_data(monkeypatch)

    click_button(app, "Delete Dataset")

    assert not app.exception
    assert len(app.metric) == 0
    assert button_labels(app) == ["Load Sample Dataset"]
    assert app.session_state["df"] is None
    assert app.session_state["ai_insights"] is None

    click_button(app, "Load Sample Dataset")
    assert {metric.label: metric.value for metric in app.metric}["Rows"] == "44"


def test_upload_delete_and_upload_again(monkeypatch):
    app = start_app(monkeypatch)

    app.file_uploader[0].upload("sales.csv", SAMPLE_CSV.read_bytes(), "text/csv").run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric}["Rows"] == "44"

    click_button(app, "Delete Dataset")
    # The uploader comes back empty, ready for a different file
    assert app.file_uploader[0].value is None

    app.file_uploader[0].upload("small.csv", b"name,amount\nAlice,1\nBob,2\n", "text/csv").run()
    assert {metric.label: metric.value for metric in app.metric}["Rows"] == "2"


def test_invalid_upload_shows_error_and_keeps_start_buttons(monkeypatch):
    app = start_app(monkeypatch)

    app.file_uploader[0].upload("broken.csv", b"a,b\n1,2\n3,4,5,6\n", "text/csv").run()

    assert not app.exception
    assert "could not be parsed" in app.error[0].value
    assert button_labels(app) == ["Load Sample Dataset"]


def test_app_works_without_gemini_key(monkeypatch):
    app = run_app_with_sample_data(monkeypatch)

    assert not app.exception
    assert any("no Gemini API key" in info.value for info in app.info)
    ai_button = next(button for button in app.button if button.label == "Generate AI Insights")
    assert ai_button.disabled


def test_generate_ai_insights_shows_result(monkeypatch):
    # Replace the real Gemini call so the test never contacts Google
    monkeypatch.setattr(ai_insights, "generate_ai_insights", lambda results: "Fake insight about 2 duplicates.")
    app = start_app(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    click_button(app, "Load Sample Dataset")

    click_button(app, "Generate AI Insights")

    assert not app.exception
    assert any("Fake insight about 2 duplicates." in block.value for block in app.markdown)
    assert app.session_state["ai_insights"] == "Fake insight about 2 duplicates."
