"""Data quality checks for CSV datasets.

Everything in this file is plain pandas. The AI feature never touches these
checks - it only receives the results afterwards and explains them.
"""

import json
from datetime import datetime, timezone

import pandas as pd

# The IQR method needs enough values to produce meaningful quartiles.
# With only 2 or 3 numbers, Q1 and Q3 are basically guesses.
MIN_VALUES_FOR_OUTLIER_CHECK = 5

# Values further than 1.5 x IQR outside the middle 50% get flagged.
# 1.5 is the conventional multiplier (it's what box plots use for whiskers).
IQR_MULTIPLIER = 1.5


def load_csv(file):
    """Read a CSV file (path or file-like object) into a DataFrame.

    Raises ValueError with a readable message when the file can't be used,
    so the app can show the message instead of a stack trace.
    """
    try:
        # utf-8-sig handles both normal UTF-8 and the invisible "BOM" marker
        # that Excel adds to the start of CSV files it exports.
        return pd.read_csv(file, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty. Please upload a CSV with a header row and data.")
    except pd.errors.ParserError:
        # Usually means some rows have more columns than the header
        raise ValueError(
            "The file could not be parsed as a CSV. Check that every row has "
            "the same number of columns and that quotes are closed properly."
        )
    except UnicodeDecodeError:
        # Happens with Excel files renamed to .csv, or non-UTF-8 text encodings
        raise ValueError("The file is not valid UTF-8 text. Please save it as a UTF-8 CSV and try again.")


def is_blank_string(value):
    """True for "" and strings that only contain spaces, tabs, etc."""
    return isinstance(value, str) and value.strip() == ""


def get_text_columns(df):
    """Columns that can contain strings (anything that isn't numeric)."""
    return [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]


def get_missing_value_mask(df):
    """Return a True/False table the same shape as df, where True means "missing".

    pandas only treats real nulls (NaN/None) as missing, but in practice a
    cell containing "" or "   " is just as empty to anyone reading a report.
    So we start with pandas' null check and then also mark blank strings.
    """
    missing_mask = df.isna()

    for column in get_text_columns(df):
        blank_strings = df[column].map(is_blank_string).astype(bool)
        missing_mask[column] = missing_mask[column] | blank_strings

    return missing_mask


def get_dataset_summary(df):
    """Basic shape of the dataset plus each column's pandas data type."""
    columns = [{"name": str(column), "data_type": str(df[column].dtype)} for column in df.columns]
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
    }


def find_missing_values(df):
    """Count missing values (nulls, empty strings, whitespace-only) per column."""
    missing_mask = get_missing_value_mask(df)

    by_column = {}
    for column in df.columns:
        by_column[str(column)] = int(missing_mask[column].sum())

    return {
        "total_missing_cells": int(missing_mask.sum().sum()),
        # A row with 3 missing cells counts once here, which is why this is
        # reported separately from the cell total.
        "rows_with_missing_values": int(missing_mask.any(axis=1).sum()),
        "by_column": by_column,
    }


def find_duplicate_rows(df):
    """Find rows that are exact copies of an earlier row.

    duplicated() marks the 2nd, 3rd, ... occurrence of a row as True and
    leaves the first one as False. That way the count is "how many extra
    copies exist", which is the number of rows you'd remove when deduplicating.
    Every column has to match exactly, so "Alice" and "Alice " are different.
    """
    is_duplicate = df.duplicated(keep="first")
    duplicate_indexes = df.index[is_duplicate].tolist()

    return {
        "duplicate_row_count": int(is_duplicate.sum()),
        "duplicate_row_indexes": [int(index) for index in duplicate_indexes],
    }


def has_extra_whitespace(value):
    """True if a string starts or ends with whitespace, like " Laptop" or "North ".

    Whitespace-only strings are skipped because they are already counted as
    missing values - we don't want the same cell showing up in two checks.
    """
    if not isinstance(value, str) or is_blank_string(value):
        return False
    return value != value.strip()


def find_whitespace_issues(df):
    """Count cells with leading or trailing whitespace in each text column."""
    by_column = {}
    for column in get_text_columns(df):
        # astype(bool) matters for empty columns: map() on an empty text
        # column keeps the text type, and summing it would return "" not 0.
        count = int(df[column].map(has_extra_whitespace).astype(bool).sum())
        if count > 0:
            by_column[str(column)] = count

    return {
        "total_cells_with_whitespace": sum(by_column.values()),
        "by_column": by_column,
    }


def find_numeric_outliers(df):
    """Flag possible outliers in numeric columns using the IQR method.

    How it works:
      - Q1 is the value 25% of the way through the sorted data, Q3 is 75%.
      - IQR = Q3 - Q1, the range covering the middle half of the values.
      - Anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is flagged.

    Because it's based on quartiles rather than the mean, one huge value
    can't drag the thresholds around much. A flagged value is only *unusual*,
    not necessarily wrong - a genuine bulk order is still a valid order.
    """
    analyzed_columns = {}
    skipped_columns = {}

    for column in df.select_dtypes(include="number").columns:
        values = df[column].dropna()

        if len(values) < MIN_VALUES_FOR_OUTLIER_CHECK:
            skipped_columns[str(column)] = (
                f"Only {len(values)} numeric values; at least "
                f"{MIN_VALUES_FOR_OUTLIER_CHECK} are needed."
            )
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        # If the middle 50% of values are all identical, IQR is 0 and every
        # value that differs even slightly would be flagged, which is noise.
        if iqr == 0:
            skipped_columns[str(column)] = "Most values are identical, so the IQR is zero."
            continue

        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        outliers = values[(values < lower_bound) | (values > upper_bound)]

        analyzed_columns[str(column)] = {
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
            "outlier_count": int(len(outliers)),
            "outlier_row_indexes": [int(index) for index in outliers.index],
        }

    total_outliers = sum(result["outlier_count"] for result in analyzed_columns.values())

    # One row can be flagged in several columns (e.g. a huge quantity also
    # makes the total huge), so count unique rows separately from values.
    flagged_rows = set()
    for result in analyzed_columns.values():
        flagged_rows.update(result["outlier_row_indexes"])

    return {
        "total_outlier_values": total_outliers,
        "rows_with_outliers": len(flagged_rows),
        "flagged_row_indexes": sorted(flagged_rows),
        "by_column": analyzed_columns,
        "skipped_columns": skipped_columns,
    }


def run_all_checks(df):
    """Run every check and collect the results in one dictionary.

    This dictionary is the single source of truth: the dashboard, the JSON
    report and the AI prompt are all built from it.
    """
    return {
        "dataset_summary": get_dataset_summary(df),
        "missing_values": find_missing_values(df),
        "duplicates": find_duplicate_rows(df),
        "whitespace": find_whitespace_issues(df),
        "outliers": find_numeric_outliers(df),
    }


def summarize_findings(results):
    """Turn the check results into short, plain-English sentences."""
    summary = results["dataset_summary"]
    missing = results["missing_values"]
    duplicates = results["duplicates"]
    whitespace = results["whitespace"]
    outliers = results["outliers"]

    if summary["row_count"] == 0:
        return ["The dataset has column headers but no rows, so there is nothing to check."]

    sentences = [f"The dataset has {summary['row_count']} rows and {summary['column_count']} columns."]

    if missing["total_missing_cells"] > 0:
        columns_affected = [name for name, count in missing["by_column"].items() if count > 0]
        sentences.append(
            f"{missing['total_missing_cells']} cells are missing or blank, spread across "
            f"{missing['rows_with_missing_values']} rows and {len(columns_affected)} columns "
            f"({', '.join(columns_affected)})."
        )
    else:
        sentences.append("No missing or blank values were found.")

    if duplicates["duplicate_row_count"] > 0:
        sentences.append(
            f"{duplicates['duplicate_row_count']} rows are exact duplicates of an earlier row."
        )
    else:
        sentences.append("No exact duplicate rows were found.")

    if whitespace["total_cells_with_whitespace"] > 0:
        sentences.append(
            f"{whitespace['total_cells_with_whitespace']} text values have extra spaces at the "
            f"start or end ({', '.join(whitespace['by_column'])}). These can make identical "
            "values look different when grouping or filtering."
        )
    else:
        sentences.append("No leading or trailing whitespace was found in text columns.")

    if outliers["total_outlier_values"] > 0:
        flagged_columns = [name for name, info in outliers["by_column"].items() if info["outlier_count"] > 0]
        sentences.append(
            f"{outliers['total_outlier_values']} numeric values in {outliers['rows_with_outliers']} rows "
            f"look unusually high or low ({', '.join(flagged_columns)}). These are possible "
            "anomalies worth reviewing, not confirmed errors."
        )
    elif outliers["by_column"]:
        sentences.append("No numeric outliers were found using the IQR method.")
    else:
        sentences.append("No numeric columns had enough data for an outlier check.")

    return sentences


def build_json_report(results, dataset_name, ai_insights=None):
    """Bundle the results into a JSON string for the download button."""
    report = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "summary": summarize_findings(results),
        **results,
        # None (null in JSON) makes it clear that AI insights weren't generated,
        # rather than leaving the reader wondering if the key was forgotten.
        "ai_insights": ai_insights,
    }
    return json.dumps(report, indent=2)
