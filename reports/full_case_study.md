# Credit Risk Analyst Case Study

## 1. Business Context

The objective is to estimate credit default risk for loan applicants using borrower, loan, and credit-history variables. This type of analysis supports underwriting, portfolio monitoring, early warning systems, and regulatory-style risk discussions.

## 2. Dataset Overview

- Source records: 32,581
- Records used after cleaning: 32,574
- Removed records: 7
- Target column: `loan_status`
- Observed default rate: 21.82%

## 3. Data Quality Audit

| column | dtype | non_null | missing | missing_pct | unique_values |
| --- | --- | --- | --- | --- | --- |
| person_age | int64 | 32574 | 0 | 0.0000 | 56 |
| person_income | int64 | 32574 | 0 | 0.0000 | 4294 |
| person_home_ownership | str | 32574 | 0 | 0.0000 | 4 |
| person_emp_length | float64 | 31679 | 895 | 2.7500 | 35 |
| loan_intent | str | 32574 | 0 | 0.0000 | 6 |
| loan_grade | str | 32574 | 0 | 0.0000 | 7 |
| loan_amnt | int64 | 32574 | 0 | 0.0000 | 753 |
| loan_int_rate | float64 | 29459 | 3115 | 9.5600 | 348 |
| loan_status | int64 | 32574 | 0 | 0.0000 | 2 |
| loan_percent_income | float64 | 32574 | 0 | 0.0000 | 77 |
| cb_person_default_on_file | str | 32574 | 0 | 0.0000 | 2 |
| cb_person_cred_hist_length | int64 | 32574 | 0 | 0.0000 | 29 |

Key cleaning choices:

- Removed unrealistic borrower ages outside 18 to 100.
- Removed unrealistic employment length above 60 years.
- Removed non-positive income or loan amount records.
- Imputed missing numeric values inside the modelling pipeline.

## 4. Numeric Variable Summary

| feature | count | mean | std | min | 1% | 5% | 25% | 50% | 75% | 95% | 99% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| person_age | 32574.0000 | 27.7184 | 6.2050 | 20.0000 | 21.0000 | 22.0000 | 23.0000 | 26.0000 | 30.0000 | 40.0000 | 50.0000 | 94.0000 |
| person_income | 32574.0000 | 65878.4808 | 52531.9388 | 4000.0000 | 14400.0000 | 22880.0000 | 38500.0000 | 55000.0000 | 79200.0000 | 138000.0000 | 225000.0000 | 2039784.0000 |
| person_emp_length | 31679.0000 | 4.7821 | 4.0349 | 0.0000 | 0.0000 | 0.0000 | 2.0000 | 4.0000 | 7.0000 | 13.0000 | 18.0000 | 41.0000 |
| loan_amnt | 32574.0000 | 9588.0181 | 6320.2496 | 500.0000 | 1000.0000 | 2000.0000 | 5000.0000 | 8000.0000 | 12200.0000 | 24000.0000 | 29727.0000 | 35000.0000 |
| loan_int_rate | 29459.0000 | 11.0115 | 3.2405 | 5.4200 | 5.4200 | 6.0300 | 7.9000 | 10.9900 | 13.4700 | 16.3200 | 18.6200 | 23.2200 |
| loan_status | 32574.0000 | 0.2182 | 0.4130 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| loan_percent_income | 32574.0000 | 0.1702 | 0.1068 | 0.0000 | 0.0200 | 0.0400 | 0.0900 | 0.1500 | 0.2300 | 0.3800 | 0.5000 | 0.8300 |
| cb_person_cred_hist_length | 32574.0000 | 5.8041 | 4.0539 | 2.0000 | 2.0000 | 2.0000 | 3.0000 | 4.0000 | 8.0000 | 14.0000 | 17.0000 | 30.0000 |

## 5. Default Rate by Loan Grade

| loan_grade | records | defaults | default_rate |
| --- | --- | --- | --- |
| G | 64 | 63 | 0.9844 |
| F | 241 | 170 | 0.7054 |
| E | 964 | 621 | 0.6442 |
| D | 3625 | 2140 | 0.5903 |
| C | 6456 | 1339 | 0.2074 |
| B | 10448 | 1701 | 0.1628 |
| A | 10776 | 1073 | 0.0996 |

## 6. Default Rate by Loan Purpose

| loan_intent | records | defaults | default_rate |
| --- | --- | --- | --- |
| DEBTCONSOLIDATION | 5212 | 1490 | 0.2859 |
| MEDICAL | 6071 | 1621 | 0.2670 |
| HOMEIMPROVEMENT | 3605 | 941 | 0.2610 |
| PERSONAL | 5519 | 1097 | 0.1988 |
| EDUCATION | 6451 | 1111 | 0.1722 |
| VENTURE | 5716 | 847 | 0.1482 |

## 7. Default Rate by Home Ownership

| person_home_ownership | records | defaults | default_rate |
| --- | --- | --- | --- |
| RENT | 16442 | 5191 | 0.3157 |
| OTHER | 107 | 33 | 0.3084 |
| MORTGAGE | 13441 | 1690 | 0.1257 |
| OWN | 2584 | 193 | 0.0747 |

## 8. Default Rate by Loan Percent Income Band

| loan_percent_income_band | records | defaults | default_rate |
| --- | --- | --- | --- |
| (-0.001, 0.08] | 7569 | 864 | 0.1141 |
| (0.08, 0.12] | 5585 | 696 | 0.1246 |
| (0.12, 0.18] | 7343 | 1114 | 0.1517 |
| (0.18, 0.25] | 5799 | 1110 | 0.1914 |
| (0.25, 0.83] | 6278 | 3323 | 0.5293 |

## 9. Model Choice

I used logistic regression because it is interpretable, stable, and easy to explain in banking interviews. The goal is not only prediction accuracy, but also understanding why borrowers are classified as higher or lower risk.

## 10. Model Validation

| Metric | Value |
| --- | ---: |
| ROC AUC | 0.8608 |
| KS Statistic | 0.5898 |
| Accuracy | 0.8371 |
| Precision | 0.6072 |
| Recall | 0.7169 |
| F1 Score | 0.6575 |

The AUC and KS values show that the model separates high-risk and low-risk borrowers well. Recall is important because missing risky borrowers can create credit losses.

## 11. Risk Bands

| risk_band | accounts | avg_pd | default_rate |
| --- | --- | --- | --- |
| A_lowest | 1629 | 0.0203 | 0.0246 |
| B_low | 1629 | 0.0530 | 0.0743 |
| C_medium | 1628 | 0.1117 | 0.0940 |
| D_high | 1629 | 0.2526 | 0.2087 |
| E_highest | 1629 | 0.6558 | 0.6894 |

Risk bands convert raw probabilities into a portfolio-monitoring view. The highest band should have a much higher default rate than the lowest band. This project shows clear rank ordering.

## 12. Top Model Drivers

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

Positive coefficients increase estimated risk. Negative coefficients reduce estimated risk. These drivers should be reviewed with business judgment before any production use.

## 13. Stress Testing

| scenario | average_pd | absolute_change | relative_change_pct |
| --- | --- | --- | --- |
| baseline | 0.2186 | 0.0000 | 0.0000 |
| mild_stress | 0.2773 | 0.0587 | 26.8461 |
| severe_stress | 0.3689 | 0.1503 | 68.7389 |

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
