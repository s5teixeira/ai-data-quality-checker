"""Smoke tests that run the Streamlit app headlessly with Streamlit's AppTest."""

from streamlit.testing.v1 import AppTest

# AppTest resolves relative paths from this test file, so point one folder up
APP_PATH = "../app.py"


def run_app_with_sample_data(monkeypatch):
    # An empty key means "not configured". load_dotenv() won't overwrite a
    # variable that's already set, so a local .env can't affect this test.
    monkeypatch.setenv("GEMINI_API_KEY", "")

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    sample_button = next(button for button in app.button if button.label == "Load Sample Dataset")
    sample_button.click().run()
    return app


def test_app_starts_with_instructions():
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()

    assert not app.exception
    assert "Load Sample Dataset" in app.info[0].value


def test_sample_dataset_is_analyzed(monkeypatch):
    app = run_app_with_sample_data(monkeypatch)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Rows"] == "44"
    assert metrics["Duplicate rows"] == "2"


def test_app_works_without_gemini_key(monkeypatch):
    app = run_app_with_sample_data(monkeypatch)

    assert not app.exception
    assert any("no Gemini API key" in info.value for info in app.info)
    ai_button = next(button for button in app.button if button.label == "Generate AI Insights")
    assert ai_button.disabled
