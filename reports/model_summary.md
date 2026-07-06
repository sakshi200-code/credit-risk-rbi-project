# Credit Risk Model Summary

## Objective

Estimate credit default risk for a retail lending portfolio and translate predictions into practical risk monitoring outputs.

## Portfolio Snapshot

- Source records: 32,581
- Records after cleaning: 32,574
- Removed outlier / invalid records: 7
- Observed default rate: 21.82%
- Target: `loan_status`
- Model: Logistic regression with preprocessing pipeline

## Data Quality Notes

- Missing employment length in source data: 2.75%
- Missing loan interest rate in source data: 9.56%
- Removed records with unrealistic age, unrealistic employment length, non-positive income, or non-positive loan amount.

## Model Validation Metrics

| Metric | Value |
|---|---:|
| ROC AUC | 0.8608 |
| KS Statistic | 0.5898 |
| Accuracy | 0.8371 |
| Precision | 0.6072 |
| Recall | 0.7169 |
| F1 Score | 0.6575 |

## Confusion Matrix

Threshold: 0.30

| | Predicted Non-default | Predicted Default |
|---|---:|---:|
| Actual Non-default | 5543 | 824 |
| Actual Default | 503 | 1274 |

## Risk Bands

| risk_band | accounts | avg_pd | default_rate |
| --- | --- | --- | --- |
| A_lowest | 1629 | 0.0203 | 0.0246 |
| B_low | 1629 | 0.0530 | 0.0743 |
| C_medium | 1628 | 0.1117 | 0.0940 |
| D_high | 1629 | 0.2526 | 0.2087 |
| E_highest | 1629 | 0.6558 | 0.6894 |

## Top Model Drivers

Positive coefficients increase predicted default risk. Negative coefficients reduce predicted default risk.

| feature | coefficient |
| --- | --- |
| loan_percent_income | 1.2736 |
| person_home_ownership_OWN | -1.1265 |
| loan_grade_C | -0.9495 |
| loan_grade_B | -0.7922 |
| loan_grade_D | 0.7340 |
| loan_intent_VENTURE | -0.6609 |
| person_home_ownership_RENT | 0.6227 |
| loan_int_rate | 0.6219 |
| loan_grade_A | -0.5600 |
| loan_grade_E | 0.5158 |
| loan_amnt | -0.5125 |
| loan_intent_EDUCATION | -0.4839 |

## Stress Testing

| scenario | average_pd | absolute_change | relative_change_pct |
| --- | --- | --- | --- |
| baseline | 0.2186 | 0.0000 | 0.0000 |
| mild_stress | 0.2773 | 0.0587 | 26.8461 |
| severe_stress | 0.3689 | 0.1503 | 68.7389 |

## Analyst Interpretation

The model ranks borrowers by credit risk with strong separation between lower and higher risk bands. Key risk drivers include loan percent of income, home ownership, loan grade, loan interest rate, loan amount, and loan purpose. The stress test shows how portfolio-level PD can rise when affordability and pricing assumptions deteriorate.

## Governance Notes

For a production banking model, the next steps would include out-of-time validation, stability testing, fairness checks, monitoring for population drift, documentation of assumptions, challenger model comparison, and periodic recalibration.
