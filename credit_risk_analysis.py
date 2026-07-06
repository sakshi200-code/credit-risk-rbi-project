"""
End-to-end credit risk analyst case study.

This file is intentionally written as a single readable analysis script for
portfolio review. The modular production-style code is inside src/, while this
script shows the complete analyst workflow in one place:

1. Load the lending dataset.
2. Audit data quality.
3. Clean outliers.
4. Create EDA summary tables.
5. Train an interpretable credit risk model.
6. Validate discriminatory power.
7. Build risk bands.
8. Run stress scenarios.
9. Export reports for interview discussion.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import TARGET, load_credit_risk_dataset
from src.model_pipeline import train_and_evaluate
from src.reporting import write_interview_notes, write_model_summary
from src.stress_testing import run_stress_tests


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
TABLE_DIR = REPORT_DIR / "tables"

ZIP_PATH = DATA_DIR / "archive.zip"
EXTRACTED_CSV_PATH = DATA_DIR / "credit_risk_dataset.csv"
MODEL_SUMMARY_PATH = REPORT_DIR / "model_summary.md"
INTERVIEW_NOTES_PATH = REPORT_DIR / "interview_notes.md"
CASE_STUDY_PATH = REPORT_DIR / "full_case_study.md"


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)


def basic_dataset_profile(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        series = data[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    numeric = data.select_dtypes(include=["number"])
    summary = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    summary = summary.reset_index().rename(columns={"index": "feature"})
    return summary.round(4)


def categorical_summary(data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = data.select_dtypes(include=["object", "string", "category"]).columns
    rows = []
    for column in categorical_columns:
        counts = data[column].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "feature": column,
                    "value": str(value),
                    "records": int(count),
                    "share_pct": round(float(count / len(data) * 100), 2),
                }
            )
    return pd.DataFrame(rows)


def default_rate_by_category(data: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        data.groupby(column, dropna=False)[TARGET]
        .agg(records="size", defaults="sum", default_rate="mean")
        .reset_index()
        .sort_values("default_rate", ascending=False)
    )
    grouped["default_rate"] = grouped["default_rate"].round(4)
    return grouped


def default_rate_by_numeric_bins(data: pd.DataFrame, column: str, bins: int = 5) -> pd.DataFrame:
    temporary = data[[column, TARGET]].dropna().copy()
    temporary[f"{column}_band"] = pd.qcut(temporary[column], q=bins, duplicates="drop")
    grouped = (
        temporary.groupby(f"{column}_band", observed=False)[TARGET]
        .agg(records="size", defaults="sum", default_rate="mean")
        .reset_index()
    )
    grouped[f"{column}_band"] = grouped[f"{column}_band"].astype(str)
    grouped["default_rate"] = grouped["default_rate"].round(4)
    return grouped


def create_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    enriched["loan_to_income_ratio"] = enriched["loan_amnt"] / enriched["person_income"].replace(0, np.nan)
    enriched["high_interest_flag"] = (enriched["loan_int_rate"] >= enriched["loan_int_rate"].median()).astype(int)
    enriched["short_credit_history_flag"] = (enriched["cb_person_cred_hist_length"] <= 3).astype(int)
    enriched["prior_default_flag"] = (enriched["cb_person_default_on_file"] == "Y").astype(int)
    return enriched


def feature_relationship_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}

    categorical_features = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]
    for feature in categorical_features:
        tables[f"default_by_{feature}"] = default_rate_by_category(data, feature)

    numeric_features = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    for feature in numeric_features:
        tables[f"default_by_{feature}_band"] = default_rate_by_numeric_bins(data, feature)

    enriched = create_derived_features(data)
    tables["derived_feature_summary"] = enriched[
        [
            "loan_to_income_ratio",
            "high_interest_flag",
            "short_credit_history_flag",
            "prior_default_flag",
            TARGET,
        ]
    ].describe().T.reset_index().rename(columns={"index": "feature"}).round(4)

    return tables


def export_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    display = frame.copy()
    if max_rows is not None:
        display = display.head(max_rows)

    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")

    headers = [str(column) for column in display.columns]
    rows = display.astype(str).values.tolist()
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def write_full_case_study(
    audit: dict,
    profile: pd.DataFrame,
    numeric: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    model_results: dict,
    stress_results: pd.DataFrame,
) -> None:
    metrics = model_results["metrics"]
    coefficients = model_results["top_coefficients"]
    risk_bands = model_results["risk_bands"]

    content = f"""# Credit Risk Analyst Case Study

## 1. Business Context

The objective is to estimate credit default risk for loan applicants using borrower, loan, and credit-history variables. This type of analysis supports underwriting, portfolio monitoring, early warning systems, and regulatory-style risk discussions.

## 2. Dataset Overview

- Source records: {audit["source_rows"]:,}
- Records used after cleaning: {audit["model_rows"]:,}
- Removed records: {audit["removed_rows"]:,}
- Target column: `{audit["target_column"]}`
- Observed default rate: {audit["default_rate"]:.2%}

## 3. Data Quality Audit

{markdown_table(profile)}

Key cleaning choices:

- Removed unrealistic borrower ages outside 18 to 100.
- Removed unrealistic employment length above 60 years.
- Removed non-positive income or loan amount records.
- Imputed missing numeric values inside the modelling pipeline.

## 4. Numeric Variable Summary

{markdown_table(numeric, max_rows=12)}

## 5. Default Rate by Loan Grade

{markdown_table(tables["default_by_loan_grade"])}

## 6. Default Rate by Loan Purpose

{markdown_table(tables["default_by_loan_intent"])}

## 7. Default Rate by Home Ownership

{markdown_table(tables["default_by_person_home_ownership"])}

## 8. Default Rate by Loan Percent Income Band

{markdown_table(tables["default_by_loan_percent_income_band"])}

## 9. Model Choice

I used logistic regression because it is interpretable, stable, and easy to explain in banking interviews. The goal is not only prediction accuracy, but also understanding why borrowers are classified as higher or lower risk.

## 10. Model Validation

| Metric | Value |
| --- | ---: |
| ROC AUC | {metrics["auc"]:.4f} |
| KS Statistic | {metrics["ks"]:.4f} |
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 Score | {metrics["f1"]:.4f} |

The AUC and KS values show that the model separates high-risk and low-risk borrowers well. Recall is important because missing risky borrowers can create credit losses.

## 11. Risk Bands

{markdown_table(risk_bands)}

Risk bands convert raw probabilities into a portfolio-monitoring view. The highest band should have a much higher default rate than the lowest band. This project shows clear rank ordering.

## 12. Top Model Drivers

{markdown_table(coefficients[["feature", "coefficient"]])}

Positive coefficients increase estimated risk. Negative coefficients reduce estimated risk. These drivers should be reviewed with business judgment before any production use.

## 13. Stress Testing

{markdown_table(stress_results)}

The stress test increases loan interest rate and loan percent of income to simulate weaker affordability and pricing conditions. The result shows how portfolio average PD changes under mild and severe assumptions.

## 14. RBI / Banking Interview Discussion

This project demonstrates the credit risk lifecycle:

- Define a default target.
- Audit and clean raw data.
- Study default rates across borrower and loan segments.
- Train an interpretable model.
- Validate ranking power using AUC and KS.
- Convert predictions into risk bands.
- Explain model drivers.
- Stress the portfolio under adverse assumptions.

## 15. Limitations

- The dataset does not include macroeconomic time series.
- The model does not include out-of-time validation because no date column is available.
- The project is educational and not a production credit decision system.
- Further work should include calibration plots, fairness checks, PSI monitoring, and challenger models.
"""
    CASE_STUDY_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_directories()

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {ZIP_PATH}")

    data, audit = load_credit_risk_dataset(ZIP_PATH, EXTRACTED_CSV_PATH)
    profile = basic_dataset_profile(data)
    numeric = numeric_summary(data)
    categorical = categorical_summary(data)
    relationship_tables = feature_relationship_tables(data)

    all_tables = {
        "dataset_profile": profile,
        "numeric_summary": numeric,
        "categorical_summary": categorical,
        **relationship_tables,
    }
    export_tables(all_tables)

    model_results = train_and_evaluate(data, target=TARGET)
    stress_results = run_stress_tests(model_results["model"], model_results["feature_columns"], data)

    write_model_summary(MODEL_SUMMARY_PATH, model_results, stress_results, audit)
    write_interview_notes(INTERVIEW_NOTES_PATH)
    write_full_case_study(audit, profile, numeric, relationship_tables, model_results, stress_results)

    print("Credit risk analysis completed.")
    print(f"Model summary: {MODEL_SUMMARY_PATH}")
    print(f"Full case study: {CASE_STUDY_PATH}")
    print(f"Exported tables: {TABLE_DIR}")


if __name__ == "__main__":
    main()
