# Credit Risk Analytics Project for RBI / Banking Roles

This project is a portfolio-ready credit risk analyst case study for a data science student applying to banking, regulatory, and RBI-adjacent roles.

It builds an end-to-end default risk model on borrower-level lending data, evaluates model quality, explains drivers of risk, creates credit risk bands, and runs stress scenarios.

## Why This Project Fits RBI / Banking Interviews

The project demonstrates:

- Credit risk modelling using borrower, loan, and credit-history variables.
- Probability of Default estimation with interpretable machine learning.
- Model validation using AUC, KS statistic, confusion matrix, precision, recall, and F1 score.
- Risk segmentation into score bands similar to portfolio monitoring.
- Stress testing under adverse affordability and pricing assumptions.
- Responsible modelling discussion: explainability, bias checks, monitoring, and governance.

The original dataset is stored as `data/archive.zip`; the pipeline extracts a modelling CSV and applies documented cleaning rules.

## Project Structure

```text
credit-risk-rbi-project/
  README.md
  requirements.txt
  run_project.py
  credit_risk_analysis.py
  src/
    data_loading.py
    data_simulation.py
    model_pipeline.py
    stress_testing.py
    reporting.py
  data/
    archive.zip
    credit_risk_dataset.csv
  reports/
    model_summary.md
    interview_notes.md
    full_case_study.md
    tables/
```

## Business Problem

A bank wants to estimate the probability that a borrower will default / become risky based on applicant and loan characteristics. The goal is to support:

- Credit underwriting decisions.
- Early warning signals for risky accounts.
- Portfolio-level risk monitoring.
- Stress testing under adverse affordability and pricing assumptions.

Target variable:

- `loan_status = 1`: risky / defaulted loan.
- `loan_status = 0`: non-default loan.

## Features

The dataset contains 32,581 loan records before cleaning, with features such as:

- Borrower profile: age, income, employment length, home ownership.
- Loan details: amount, interest rate, intent, grade, loan percent of income.
- Credit behaviour: prior default on file and credit history length.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full project:

```bash
python run_project.py
```

Run the detailed analyst case-study script:

```bash
python credit_risk_analysis.py
```

This creates:

- `data/credit_risk_dataset.csv`
- `reports/model_summary.md`
- `reports/interview_notes.md`
- `reports/full_case_study.md`
- `reports/tables/*.csv`

## Methods Used

The model uses a lightweight logistic regression implementation with preprocessing:

- Numeric variables are median-imputed and standardized.
- Categorical variables are imputed and one-hot encoded.
- Logistic regression is used because credit risk teams often value interpretability.
- The implementation uses only NumPy and Pandas, so it is easy to run on most student laptops.

Metrics reported:

- ROC AUC
- KS statistic
- Accuracy, precision, recall, F1 score
- Confusion matrix
- Top model coefficients
- Risk band default rates
- Stress scenario PD changes

## Main Files To Show Recruiters

- `credit_risk_analysis.py`: full readable case-study workflow.
- `src/model_pipeline.py`: reusable modelling code.
- `reports/full_case_study.md`: final analyst report.
- `reports/model_summary.md`: short model validation summary.

## Resume Bullet

Built an end-to-end credit risk modelling project using Python, logistic regression, model validation metrics, risk banding, and stress testing on borrower-level lending data; produced explainable outputs suitable for banking and RBI-oriented analyst interviews.

## Interview Pitch

This project shows that I understand the full credit risk lifecycle: defining default, preparing borrower-level data, estimating default risk, validating discriminatory power, converting predictions into risk bands, interpreting drivers, and testing portfolio sensitivity under stress. I chose logistic regression to prioritize explainability, which is important in banking and regulatory environments.

## Disclaimer

This is an educational project. It is not an official RBI model, regulatory submission, or production credit decisioning system.
