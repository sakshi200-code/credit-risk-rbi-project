from pathlib import Path

import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def write_model_summary(path: Path, results: dict, stress_results: pd.DataFrame, audit: dict) -> None:
    metrics = results["metrics"]
    confusion = metrics["confusion_matrix"]

    content = f"""# Credit Risk Model Summary

## Objective

Estimate credit default risk for a retail lending portfolio and translate predictions into practical risk monitoring outputs.

## Portfolio Snapshot

- Source records: {audit["source_rows"]:,}
- Records after cleaning: {audit["model_rows"]:,}
- Removed outlier / invalid records: {audit["removed_rows"]:,}
- Observed default rate: {metrics["default_rate"]:.2%}
- Target: `{audit["target_column"]}`
- Model: Logistic regression with preprocessing pipeline

## Data Quality Notes

- Missing employment length in source data: {audit["missing_person_emp_length_pct"]:.2f}%
- Missing loan interest rate in source data: {audit["missing_loan_int_rate_pct"]:.2f}%
- Removed records with unrealistic age, unrealistic employment length, non-positive income, or non-positive loan amount.

## Model Validation Metrics

| Metric | Value |
|---|---:|
| ROC AUC | {metrics["auc"]:.4f} |
| KS Statistic | {metrics["ks"]:.4f} |
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 Score | {metrics["f1"]:.4f} |

## Confusion Matrix

Threshold: {metrics["classification_threshold"]:.2f}

| | Predicted Non-default | Predicted Default |
|---|---:|---:|
| Actual Non-default | {confusion[0][0]} | {confusion[0][1]} |
| Actual Default | {confusion[1][0]} | {confusion[1][1]} |

## Risk Bands

{_markdown_table(results["risk_bands"])}

## Top Model Drivers

Positive coefficients increase predicted default risk. Negative coefficients reduce predicted default risk.

{_markdown_table(results["top_coefficients"][["feature", "coefficient"]])}

## Stress Testing

{_markdown_table(stress_results)}

## Analyst Interpretation

The model ranks borrowers by credit risk with strong separation between lower and higher risk bands. Key risk drivers include loan percent of income, home ownership, loan grade, loan interest rate, loan amount, and loan purpose. The stress test shows how portfolio-level PD can rise when affordability and pricing assumptions deteriorate.

## Governance Notes

For a production banking model, the next steps would include out-of-time validation, stability testing, fairness checks, monitoring for population drift, documentation of assumptions, challenger model comparison, and periodic recalibration.
"""
    path.write_text(content, encoding="utf-8")


def write_interview_notes(path: Path) -> None:
    content = """# Interview Notes for RBI / Banking Credit Risk Roles

## 60-Second Explanation

I built an end-to-end credit risk project that estimates borrower default risk using a real lending dataset. I cleaned borrower and loan variables, handled missing values, trained an interpretable logistic regression model, validated it using AUC and KS, created risk bands, and ran portfolio stress scenarios.

## Why Logistic Regression?

Logistic regression is transparent, fast, and widely understood in credit risk. Even when stronger machine learning models are available, banks and regulators often care about interpretability, stability, monitoring, and clear explanation of risk drivers.

## Key Credit Risk Concepts Used

- PD: Probability of Default.
- Risk banding: Grouping borrowers into ordered risk grades.
- KS statistic: Measures separation between good and bad accounts.
- AUC: Measures ranking power across thresholds.
- Stress testing: Checks portfolio sensitivity under adverse borrower and pricing assumptions.
- Model governance: Documentation, validation, monitoring, and recalibration.

## How I Would Improve It

- Add reject inference if rejected applicants are available.
- Compare logistic regression with random forest, XGBoost, and scorecard binning.
- Add calibration plots and expected calibration error.
- Track population stability index over time.
- Add fairness analysis across borrower segments.
- Use real public datasets such as LendingClub or German Credit for a second version.

## Resume Bullet

Built a credit risk modelling pipeline in Python using borrower-level lending data, logistic regression, AUC/KS validation, risk banding, coefficient-based explainability, data-quality checks, and stress testing for banking and RBI-oriented analyst interviews.
"""
    path.write_text(content, encoding="utf-8")
