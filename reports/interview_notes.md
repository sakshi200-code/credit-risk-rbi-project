# Interview Notes for RBI / Banking Credit Risk Roles

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
