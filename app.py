"""Streamlit dashboard for the AI Data Quality Checker.

Run with:  streamlit run app.py

Streamlit re-runs this whole script from top to bottom every time the user
clicks something. st.session_state is how we remember things (like which
dataset is selected) between those re-runs.
"""

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import ai_insights
import data_quality

# Reads GEMINI_API_KEY / GEMINI_MODEL from a local .env file if one exists.
# Without a .env file this does nothing and the AI section stays disabled.
load_dotenv()

SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data" / "sample_sales.csv"
MAX_UPLOAD_MB = 10
PREVIEW_ROWS = 100


def select_sample_dataset():
    st.session_state.use_sample = True
    st.session_state.ai_insights = None


def select_uploaded_file():
    st.session_state.use_sample = False
    st.session_state.ai_insights = None


def load_selected_dataset(uploaded_file):
    """Return (dataframe, dataset_name), or (None, None) if nothing is selected yet."""
    if st.session_state.use_sample:
        return data_quality.load_csv(SAMPLE_DATA_PATH), SAMPLE_DATA_PATH.name

    if uploaded_file is not None:
        # getvalue() always returns the full file, even if it was read on a previous re-run
        return data_quality.load_csv(io.BytesIO(uploaded_file.getvalue())), uploaded_file.name

    return None, None


def show_metric_cards(results):
    summary = results["dataset_summary"]
    columns = st.columns(6)
    columns[0].metric("Rows", summary["row_count"], border=True)
    columns[1].metric("Columns", summary["column_count"], border=True)
    columns[2].metric("Missing cells", results["missing_values"]["total_missing_cells"], border=True)
    columns[3].metric("Duplicate rows", results["duplicates"]["duplicate_row_count"], border=True)
    columns[4].metric("Whitespace issues", results["whitespace"]["total_cells_with_whitespace"], border=True)
    columns[5].metric("Possible outliers", results["outliers"]["total_outlier_values"], border=True)


def show_overview(results):
    st.subheader("Summary")
    for sentence in data_quality.summarize_findings(results):
        st.markdown(f"- {sentence}")

    st.subheader("Missing values by column")
    missing_by_column = results["missing_values"]["by_column"]
    chart_data = pd.DataFrame(
        [{"column": name, "missing_values": count} for name, count in missing_by_column.items() if count > 0]
    )

    if chart_data.empty:
        st.success("No missing values found in any column.")
        return

    st.bar_chart(
        chart_data,
        x="column",
        y="missing_values",
        horizontal=True,
        sort="-missing_values",
        x_label="Missing or blank cells",
        y_label="",
    )
    st.caption("Counts include empty cells, blank text and whitespace-only text. Columns with no missing values are not shown.")


def show_missing_values(df, results):
    missing = results["missing_values"]
    st.markdown("#### Missing values")
    st.write(
        f"{missing['total_missing_cells']} missing cells across "
        f"{missing['rows_with_missing_values']} rows."
    )

    table = pd.DataFrame(
        {
            "column": list(missing["by_column"].keys()),
            "missing_cells": list(missing["by_column"].values()),
        }
    )
    table["percent_missing"] = (table["missing_cells"] / len(df) * 100).round(1)
    st.dataframe(table, hide_index=True)


def show_duplicates(df, results):
    duplicates = results["duplicates"]
    st.markdown("#### Duplicate rows")

    if duplicates["duplicate_row_count"] == 0:
        st.success("No exact duplicate rows found.")
        return

    st.write(f"{duplicates['duplicate_row_count']} rows are exact copies of an earlier row.")
    # keep=False marks every copy (including the first), so the user sees
    # each original next to its duplicate.
    st.dataframe(df[df.duplicated(keep=False)])
    st.caption("The left-most number is the row index. The first copy of each row is kept as the original.")


def show_whitespace_issues(df, results):
    whitespace = results["whitespace"]
    st.markdown("#### Leading or trailing whitespace")

    if whitespace["total_cells_with_whitespace"] == 0:
        st.success("No leading or trailing whitespace found.")
        return

    rows = []
    for column, count in whitespace["by_column"].items():
        affected = df[column][df[column].map(data_quality.has_extra_whitespace).astype(bool)]
        # Quotes make the extra spaces visible in the table
        examples = [f'"{value}"' for value in affected.unique()[:3]]
        rows.append({"column": column, "affected_cells": count, "examples": ", ".join(examples)})

    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Whitespace-only values are counted as missing values instead, so no cell is counted twice.")


def show_outliers(df, results):
    outliers = results["outliers"]
    st.markdown("#### Possible numeric outliers (IQR method)")
    st.write(
        "Values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR are flagged. These are "
        "**possible anomalies** that are worth a second look. They are not automatically wrong."
    )

    if outliers["by_column"]:
        rows = []
        for column, info in outliers["by_column"].items():
            rows.append(
                {
                    "column": column,
                    "lower_bound": info["lower_bound"],
                    "upper_bound": info["upper_bound"],
                    "flagged_values": info["outlier_count"],
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("No numeric columns could be checked for outliers.")

    for column, reason in outliers["skipped_columns"].items():
        st.caption(f"Skipped **{column}**: {reason}")

    if outliers["total_outlier_values"] > 0:
        st.write(f"Rows containing at least one flagged value ({outliers['rows_with_outliers']}):")
        st.dataframe(df.loc[outliers["flagged_row_indexes"]])


def show_data_preview(df, results):
    st.subheader("Original data")
    if len(df) > PREVIEW_ROWS:
        st.caption(f"Showing the first {PREVIEW_ROWS} of {len(df)} rows. The data is never modified.")
    st.dataframe(df.head(PREVIEW_ROWS))

    st.subheader("Columns and data types")
    column_table = pd.DataFrame(results["dataset_summary"]["columns"])
    column_table["missing_cells"] = column_table["name"].map(results["missing_values"]["by_column"])
    st.dataframe(column_table, hide_index=True)


def show_ai_section(results):
    st.subheader("AI insights")
    st.write(
        "Get a plain-English explanation of the findings, why they matter, and what to fix first. "
        "This is optional. All the checks above are done by pandas, not AI."
    )

    with st.expander("What will be shared with Google Gemini?"):
        st.write(
            "Only the aggregated results below: row and column counts, column names and data types, "
            "and issue counts per column. **No rows or cell values from your file are sent.**"
        )
        st.write(
            "On the Gemini API free tier, Google may use submitted content to improve its products, "
            "as described in Google's terms. Don't generate insights if your column names "
            "themselves are confidential."
        )
        st.json(ai_insights.build_ai_payload(results))

    if not ai_insights.is_ai_configured():
        st.info(
            "AI insights are turned off because no Gemini API key is configured. "
            "Everything else in the app works without one.\n\n"
            "To enable it, create a free API key in Google AI Studio, copy `.env.example` "
            "to `.env`, set `GEMINI_API_KEY`, and restart the app."
        )
        st.button("Generate AI Insights", disabled=True)
        return

    st.caption(f"Model: `{ai_insights.get_model_name()}`")
    if st.button("Generate AI Insights", type="primary"):
        with st.spinner("Asking Gemini to explain the findings..."):
            try:
                st.session_state.ai_insights = ai_insights.generate_ai_insights(results)
            except RuntimeError as error:
                st.error(str(error))

    if st.session_state.ai_insights:
        st.markdown(st.session_state.ai_insights)
        st.caption("AI-generated text. The numbers come from the pandas checks; review the advice before acting on it.")


def main():
    st.set_page_config(page_title="AI Data Quality Checker", layout="wide")

    st.session_state.setdefault("use_sample", False)
    st.session_state.setdefault("ai_insights", None)

    st.title("AI Data Quality Checker")
    st.write(
        "Upload a CSV file to check it for missing values, duplicate rows, extra whitespace "
        "and unusual numbers. Your file is analyzed as-is and is never modified."
    )

    upload_column, sample_column = st.columns([3, 1], vertical_alignment="center")
    with upload_column:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type="csv",
            max_upload_size=MAX_UPLOAD_MB,
            on_change=select_uploaded_file,
        )
    with sample_column:
        st.button("Load Sample Dataset", on_click=select_sample_dataset, width="stretch")
        st.caption("A fictional sales dataset with some deliberate problems.")

    try:
        df, dataset_name = load_selected_dataset(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    if df is None:
        st.info("Upload a CSV file or click **Load Sample Dataset** to get started.")
        st.stop()

    if df.empty:
        st.warning("This file has column headers but no data rows, so there is nothing to check.")
        st.stop()

    results = data_quality.run_all_checks(df)

    st.divider()
    st.markdown(f"### Results for `{dataset_name}`")
    show_metric_cards(results)

    overview_tab, findings_tab, preview_tab, ai_tab = st.tabs(
        ["Overview", "Detailed findings", "Data preview", "AI insights"]
    )
    with overview_tab:
        show_overview(results)
    with findings_tab:
        show_missing_values(df, results)
        show_duplicates(df, results)
        show_whitespace_issues(df, results)
        show_outliers(df, results)
    with preview_tab:
        show_data_preview(df, results)
    with ai_tab:
        show_ai_section(results)

    # The download button comes last so the report includes AI insights
    # generated earlier in this same run.
    st.divider()
    report_json = data_quality.build_json_report(results, dataset_name, st.session_state.ai_insights)
    st.download_button(
        "Download report (JSON)",
        data=report_json,
        file_name=f"{Path(dataset_name).stem}_quality_report.json",
        mime="application/json",
        type="primary",
    )


main()
