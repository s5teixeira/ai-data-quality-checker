import io
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import ai_insights
import data_quality

# Reads GEMINI_API_KEY / GEMINI_MODEL from a local .env file if one exists.
load_dotenv()

SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data" / "sample_sales.csv"
MAX_UPLOAD_MB = 10
PREVIEW_ROWS = 100

# Most of the theme lives in .streamlit/config.toml. These colors are only
# repeated here for the chart hover effect and the few CSS tweaks below.
BAR_COLOR = "#F97316"
BAR_HOVER_COLOR = "#FDBA74"
TABLE_STRIPE_COLOR = "#1E1F29"
ISSUE_COUNT_COLOR = "#FB923C"

# Small CSS additions for things Streamlit's theme settings can't control yet.
CUSTOM_CSS = """
<style>
/* KPI cards and expanders use a slightly lighter "card" color than the page */
[data-testid="stMetric"], [data-testid="stExpander"] details {
    background-color: #242631;
}
[data-testid="stCaptionContainer"] {
    color: #A1A1AA;
}
/* Dark text on orange buttons reads better than white; lighter orange on hover */
button[kind="primary"] {
    color: #15161D;
}
button[kind="primary"]:hover {
    background-color: #F97316;
    border-color: #F97316;
    color: #15161D;
}
/* Slightly larger tabs with a bigger click area */
[data-testid="stTabs"] [role="tab"] {
    padding: 0.6rem 1.1rem;
}
[data-testid="stTabs"] [role="tab"] p {
    font-size: 1.1rem;
}
</style>
"""


# The functions below are button/upload callbacks. Streamlit runs a callback
# *before* re-running the script, so the page always draws the new state.

def set_active_dataset(df, dataset_name):
    if df.empty:
        st.session_state.load_error = "This file has column headers but no data rows, so there is nothing to check."
        return
    st.session_state.df = df
    st.session_state.dataset_name = dataset_name
    st.session_state.ai_insights = None
    st.session_state.load_error = None


def load_sample_dataset():
    set_active_dataset(data_quality.load_csv(SAMPLE_DATA_PATH), SAMPLE_DATA_PATH.name)


def get_uploader_key():
    return f"csv_uploader_{st.session_state.uploader_version}"


def load_uploaded_file():
    uploaded_file = st.session_state[get_uploader_key()]
    if uploaded_file is None:
        return
    try:
        df = data_quality.load_csv(io.BytesIO(uploaded_file.getvalue()))
    except ValueError as error:
        st.session_state.load_error = str(error)
        return
    # The DataFrame is kept in session_state because the uploader is hidden
    # once a dataset is loaded, and a hidden widget loses its file.
    set_active_dataset(df, uploaded_file.name)


def delete_dataset():
    """Forget the loaded dataset. This never touches files on disk."""
    st.session_state.df = None
    st.session_state.dataset_name = None
    st.session_state.ai_insights = None
    st.session_state.load_error = None
    # A new key gives Streamlit a brand-new, empty file uploader
    st.session_state.uploader_version += 1


def show_dataset_picker():
    """The two starting buttons: Upload CSV (opens a small popover) and Load Sample Dataset."""
    upload_column, sample_column, _ = st.columns([1, 1, 2])
    with upload_column:
        with st.popover("Upload CSV", icon=":material/upload:", width="stretch"):
            st.file_uploader(
                "Choose a CSV file",
                type="csv",
                max_upload_size=MAX_UPLOAD_MB,
                key=get_uploader_key(),
                on_change=load_uploaded_file,
            )
    with sample_column:
        st.button(
            "Load Sample Dataset",
            icon=":material/table_view:",
            on_click=load_sample_dataset,
            width="stretch",
        )

    if st.session_state.load_error:
        st.error(st.session_state.load_error)


def style_summary_table(table, count_column):
    """Shade every other row and show counts above zero in orange.

    Only used on the small summary tables the app builds itself. The original
    data is always shown unstyled.
    """

    def stripe_every_other_row(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        styles.iloc[1::2] = f"background-color: {TABLE_STRIPE_COLOR}"
        return styles

    def highlight_issue_count(value):
        return f"color: {ISSUE_COUNT_COLOR}" if value > 0 else ""

    return (
        table.style.apply(stripe_every_other_row, axis=None)
        .map(highlight_issue_count, subset=[count_column])
        # Show values exactly as calculated (styled tables would otherwise pad decimals)
        .format(str)
    )


def show_metric_cards(results):
    summary = results["dataset_summary"]
    columns = st.columns(6)
    columns[0].metric("Rows", summary["row_count"], border=True)
    columns[1].metric("Columns", summary["column_count"], border=True)
    columns[2].metric("Missing cells", results["missing_values"]["total_missing_cells"], border=True)
    columns[3].metric("Duplicate rows", results["duplicates"]["duplicate_row_count"], border=True)
    columns[4].metric("Whitespace issues", results["whitespace"]["total_cells_with_whitespace"], border=True)
    columns[5].metric("Possible outliers", results["outliers"]["total_outlier_values"], border=True)


def build_missing_values_chart(chart_data):
    """Horizontal bar chart of missing values per column, largest first.

    This uses Altair (which comes with Streamlit) instead of st.bar_chart
    because st.bar_chart can't change a bar's color when you hover over it.
    """
    # empty=False means "no bars are highlighted until the pointer is over one"
    hover = alt.selection_point(on="pointerover", clear="pointerout", empty=False)

    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("missing_values:Q", title="Missing or blank cells", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("column:N", title=None, sort="-x"),
            color=alt.condition(hover, alt.value(BAR_HOVER_COLOR), alt.value(BAR_COLOR)),
            tooltip=[
                alt.Tooltip("column:N", title="Column"),
                alt.Tooltip("missing_values:Q", title="Missing cells"),
            ],
        )
        .add_params(hover)
    )


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

    st.altair_chart(build_missing_values_chart(chart_data), width="stretch")
    st.caption("Counts include empty cells, blank text and whitespace-only text. Columns with no missing values are not shown.")


def show_missing_values(df, results):
    missing = results["missing_values"]
    st.markdown("#### :orange[Missing values]")
    st.markdown(
        f":orange[**{missing['total_missing_cells']}**] missing cells across "
        f":orange[**{missing['rows_with_missing_values']}**] rows."
    )

    table = pd.DataFrame(
        {
            "column": list(missing["by_column"].keys()),
            "missing_cells": list(missing["by_column"].values()),
        }
    )
    table["percent_missing"] = (table["missing_cells"] / len(df) * 100).round(1)
    st.dataframe(style_summary_table(table, "missing_cells"), hide_index=True)


def show_duplicates(df, results):
    duplicates = results["duplicates"]
    st.markdown("#### :orange[Duplicate rows]")

    if duplicates["duplicate_row_count"] == 0:
        st.success("No exact duplicate rows found.")
        return

    st.markdown(f":orange[**{duplicates['duplicate_row_count']}**] rows are exact copies of an earlier row.")
    # keep=False marks every copy (including the first), so the user sees
    # each original next to its duplicate.
    st.dataframe(df[df.duplicated(keep=False)])
    st.caption("The left-most number is the row index. The first copy of each row is kept as the original.")


def show_whitespace_issues(df, results):
    whitespace = results["whitespace"]
    st.markdown("#### :orange[Leading or trailing whitespace]")

    if whitespace["total_cells_with_whitespace"] == 0:
        st.success("No leading or trailing whitespace found.")
        return

    rows = []
    for column, count in whitespace["by_column"].items():
        affected = df[column][df[column].map(data_quality.has_extra_whitespace).astype(bool)]
        # Quotes make the extra spaces visible in the table
        examples = [f'"{value}"' for value in affected.unique()[:3]]
        rows.append({"column": column, "affected_cells": count, "examples": ", ".join(examples)})

    st.dataframe(style_summary_table(pd.DataFrame(rows), "affected_cells"), hide_index=True)
    st.caption("Whitespace-only values are counted as missing values instead, so no cell is counted twice.")


def show_explore_data(df, results):
    st.subheader(":orange[Original data]")
    if len(df) > PREVIEW_ROWS:
        st.caption(f"Showing the first {PREVIEW_ROWS} of {len(df)} rows. The data is never modified.")
    st.dataframe(df.head(PREVIEW_ROWS))

    st.subheader(":orange[Columns and data types]")
    column_table = pd.DataFrame(results["dataset_summary"]["columns"])
    column_table["missing_cells"] = column_table["name"].map(results["missing_values"]["by_column"])
    st.dataframe(style_summary_table(column_table, "missing_cells"), hide_index=True)


def show_ai_section(results):
    st.subheader("AI Insights")

    if not ai_insights.is_ai_configured():
        st.info(
            "AI insights are turned off because no Gemini API key is configured. "
            "Everything else in the app works without one.\n\n"
            "To enable it, create a free API key in Google AI Studio, copy `.env.example` "
            "to `.env`, set `GEMINI_API_KEY`, and restart the app."
        )
        st.button("Generate AI Insights", disabled=True)
        return

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
    st.html(CUSTOM_CSS)

    st.session_state.setdefault("df", None)
    st.session_state.setdefault("dataset_name", None)
    st.session_state.setdefault("ai_insights", None)
    st.session_state.setdefault("load_error", None)
    st.session_state.setdefault("uploader_version", 0)

    st.title("AI Data Quality Checker")
    st.write(
        "Upload a CSV file to check it for missing values, duplicate rows, extra whitespace "
        "and unusual numbers. Your file is analyzed as-is and is never modified."
    )

    # No dataset yet: show only the two starting buttons and stop here
    if st.session_state.df is None:
        show_dataset_picker()
        st.stop()

    df = st.session_state.df
    dataset_name = st.session_state.dataset_name
    results = data_quality.run_all_checks(df)

    st.divider()
    title_column, delete_column = st.columns([4, 1], vertical_alignment="center")
    title_column.markdown(f"### Results for `{dataset_name}`")
    delete_column.button("Delete Dataset", icon=":material/delete:", on_click=delete_dataset, width="stretch")
    show_metric_cards(results)

    overview_tab, findings_tab, explore_tab, ai_tab = st.tabs(
        ["Overview", "Detailed Findings", "Explore Data", "AI Insights"]
    )
    with overview_tab:
        show_overview(results)
    with findings_tab:
        show_missing_values(df, results)
        show_duplicates(df, results)
        show_whitespace_issues(df, results)
    with explore_tab:
        show_explore_data(df, results)
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
