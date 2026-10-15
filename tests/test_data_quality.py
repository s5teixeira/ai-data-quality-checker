import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_quality

SAMPLE_CSV = Path(__file__).parent.parent / "sample_data" / "sample_sales.csv"


def make_clean_df():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Cara", "Dan", "Eve"],
            "city": ["Paris", "Rome", "Oslo", "Lima", "Kyiv"],
            "amount": [10.0, 12.0, 11.0, 13.0, 12.0],
        }
    )


# --- Clean data ---


def test_clean_dataset_has_no_issues():
    results = data_quality.run_all_checks(make_clean_df())

    assert results["dataset_summary"]["row_count"] == 5
    assert results["dataset_summary"]["column_count"] == 3
    assert results["missing_values"]["total_missing_cells"] == 0
    assert results["duplicates"]["duplicate_row_count"] == 0
    assert results["whitespace"]["total_cells_with_whitespace"] == 0
    assert results["outliers"]["total_outlier_values"] == 0


def test_dataset_summary_lists_column_types():
    summary = data_quality.get_dataset_summary(make_clean_df())

    column_names = [column["name"] for column in summary["columns"]]
    assert column_names == ["name", "city", "amount"]
    assert summary["columns"][2]["data_type"] == "float64"


# --- Missing values ---


def test_detects_null_values():
    df = pd.DataFrame({"name": ["Alice", None, "Cara"], "amount": [1.0, 2.0, np.nan]})

    missing = data_quality.find_missing_values(df)

    assert missing["by_column"] == {"name": 1, "amount": 1}
    assert missing["total_missing_cells"] == 2
    assert missing["rows_with_missing_values"] == 2


def test_empty_and_whitespace_only_strings_count_as_missing():
    df = pd.DataFrame({"name": ["Alice", "", "   ", "\t"]})

    missing = data_quality.find_missing_values(df)

    assert missing["by_column"]["name"] == 3


def test_row_with_several_missing_cells_is_counted_once():
    df = pd.DataFrame({"a": [None, "x"], "b": [None, "y"], "c": ["", "z"]})

    missing = data_quality.find_missing_values(df)

    assert missing["total_missing_cells"] == 3
    assert missing["rows_with_missing_values"] == 1


# --- Duplicates ---


def test_detects_exact_duplicate_rows():
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 2, 3, 3, 3],
            "product": ["Pen", "Ink", "Ink", "Pad", "Pad", "Pad"],
        }
    )

    duplicates = data_quality.find_duplicate_rows(df)

    # The first copy of each row is the original; only the extra copies count
    assert duplicates["duplicate_row_count"] == 3
    assert duplicates["duplicate_row_indexes"] == [2, 4, 5]


def test_rows_that_differ_only_by_whitespace_are_not_exact_duplicates():
    df = pd.DataFrame({"name": ["Alice", "Alice "], "amount": [5, 5]})

    assert data_quality.find_duplicate_rows(df)["duplicate_row_count"] == 0


# --- Whitespace ---


def test_detects_leading_and_trailing_whitespace():
    df = pd.DataFrame({"region": [" North", "South ", " East ", "West"], "amount": [1, 2, 3, 4]})

    whitespace = data_quality.find_whitespace_issues(df)

    assert whitespace["by_column"] == {"region": 3}
    assert whitespace["total_cells_with_whitespace"] == 3


def test_whitespace_only_values_are_not_double_counted():
    df = pd.DataFrame({"region": ["North", "   "]})

    results = data_quality.run_all_checks(df)

    assert results["missing_values"]["total_missing_cells"] == 1
    assert results["whitespace"]["total_cells_with_whitespace"] == 0


# --- Outliers ---


def test_detects_numeric_outlier_with_iqr():
    df = pd.DataFrame({"quantity": [2, 3, 3, 4, 4, 5, 5, 6, 250]})

    outliers = data_quality.find_numeric_outliers(df)
    quantity = outliers["by_column"]["quantity"]

    assert quantity["outlier_count"] == 1
    assert quantity["outlier_row_indexes"] == [8]
    assert quantity["q1"] == 3.0
    assert quantity["q3"] == 5.0
    assert quantity["upper_bound"] == 8.0


def test_outlier_check_skips_columns_with_too_few_values():
    df = pd.DataFrame({"price": [1.0, 2.0, None, None, 500.0]})

    outliers = data_quality.find_numeric_outliers(df)

    assert "price" not in outliers["by_column"]
    assert "price" in outliers["skipped_columns"]


def test_outlier_check_skips_columns_with_zero_iqr():
    df = pd.DataFrame({"discount": [0, 0, 0, 0, 0, 0, 10]})

    outliers = data_quality.find_numeric_outliers(df)

    assert "discount" in outliers["skipped_columns"]
    assert outliers["total_outlier_values"] == 0


def test_text_columns_are_ignored_by_outlier_check():
    df = pd.DataFrame({"name": ["a", "b", "c", "d", "e", "f"]})

    outliers = data_quality.find_numeric_outliers(df)

    assert outliers["by_column"] == {}
    assert outliers["skipped_columns"] == {}


def test_row_flagged_in_two_columns_is_one_row():
    df = pd.DataFrame(
        {
            "quantity": [1, 2, 2, 3, 3, 4, 100],
            "total": [10, 20, 20, 30, 30, 40, 1000],
        }
    )

    outliers = data_quality.find_numeric_outliers(df)

    assert outliers["total_outlier_values"] == 2
    assert outliers["rows_with_outliers"] == 1


# --- Small and empty datasets ---


def test_dataset_with_headers_but_no_rows():
    df = pd.DataFrame({"name": pd.Series(dtype="str"), "amount": pd.Series(dtype="float64")})

    results = data_quality.run_all_checks(df)

    assert results["dataset_summary"]["row_count"] == 0
    assert results["missing_values"]["total_missing_cells"] == 0
    assert results["duplicates"]["duplicate_row_count"] == 0
    assert "amount" in results["outliers"]["skipped_columns"]
    assert "no rows" in data_quality.summarize_findings(results)[0]


def test_single_row_dataset():
    df = pd.DataFrame({"name": ["Alice"], "amount": [42.0]})

    results = data_quality.run_all_checks(df)

    assert results["dataset_summary"]["row_count"] == 1
    assert results["duplicates"]["duplicate_row_count"] == 0
    assert "amount" in results["outliers"]["skipped_columns"]


# --- Loading CSV files ---


def test_load_csv_reads_valid_file():
    csv_file = io.BytesIO(b"name,amount\nAlice,10\n Bob ,20\n")

    df = data_quality.load_csv(csv_file)

    assert list(df.columns) == ["name", "amount"]
    # Whitespace must be kept, otherwise the whitespace check can't see it
    assert df["name"].tolist() == ["Alice", " Bob "]


def test_load_csv_rejects_empty_file():
    with pytest.raises(ValueError, match="empty"):
        data_quality.load_csv(io.BytesIO(b""))


def test_load_csv_rejects_malformed_file():
    malformed = io.BytesIO(b"a,b\n1,2\n3,4,5,6\n")

    with pytest.raises(ValueError, match="could not be parsed"):
        data_quality.load_csv(malformed)


def test_load_csv_rejects_non_text_file():
    binary_data = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\xff\xfe\xfa")

    with pytest.raises(ValueError):
        data_quality.load_csv(binary_data)


# --- Sample dataset and report ---


def test_sample_dataset_contains_the_intended_issues():
    df = data_quality.load_csv(SAMPLE_CSV)

    results = data_quality.run_all_checks(df)

    assert results["missing_values"]["total_missing_cells"] > 0
    assert results["duplicates"]["duplicate_row_count"] == 2
    assert results["whitespace"]["total_cells_with_whitespace"] > 0
    assert results["outliers"]["by_column"]["quantity"]["outlier_count"] >= 1


def test_json_report_is_valid_json():
    results = data_quality.run_all_checks(data_quality.load_csv(SAMPLE_CSV))

    report = json.loads(data_quality.build_json_report(results, "sample_sales.csv"))

    assert report["dataset_name"] == "sample_sales.csv"
    assert report["duplicates"]["duplicate_row_count"] == 2
    assert report["ai_insights"] is None
    assert len(report["summary"]) > 0
