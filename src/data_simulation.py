import numpy as np
import pandas as pd


def generate_credit_risk_data(n_rows: int = 8000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 66, n_rows)
    annual_income = rng.lognormal(mean=11.0, sigma=0.55, size=n_rows).round(0)
    annual_income = np.clip(annual_income, 120000, 3500000)

    employment_length = np.clip(
        rng.normal(loc=(age - 21) / 4, scale=3.0, size=n_rows), 0, 35
    ).round(1)

    credit_score = np.clip(
        rng.normal(loc=690, scale=75, size=n_rows)
        + (annual_income - annual_income.mean()) / annual_income.std() * 18,
        300,
        900,
    ).round(0)

    loan_amount = rng.lognormal(mean=12.0, sigma=0.65, size=n_rows).round(0)
    loan_amount = np.clip(loan_amount, 25000, 4500000)

    debt_to_income = np.clip(
        loan_amount / np.maximum(annual_income, 1) + rng.normal(0.18, 0.12, n_rows),
        0.02,
        1.8,
    ).round(3)

    utilization_rate = np.clip(
        rng.beta(a=2.2, b=3.0, size=n_rows) + debt_to_income / 5,
        0.01,
        1.0,
    ).round(3)

    missed_payments_6m = rng.poisson(
        lam=np.clip(0.25 + debt_to_income * 0.8 + utilization_rate * 0.7, 0.05, 4.0)
    )
    missed_payments_6m = np.clip(missed_payments_6m, 0, 6)

    loan_tenure_months = rng.choice([12, 24, 36, 48, 60, 72], n_rows, p=[0.08, 0.16, 0.3, 0.22, 0.18, 0.06])
    secured_loan = rng.choice(["secured", "unsecured"], n_rows, p=[0.42, 0.58])
    loan_purpose = rng.choice(
        ["home_improvement", "vehicle", "education", "business", "personal", "medical"],
        n_rows,
        p=[0.16, 0.2, 0.12, 0.18, 0.26, 0.08],
    )
    region = rng.choice(["north", "south", "east", "west", "central"], n_rows)

    unemployment_rate = np.clip(rng.normal(6.8, 1.2, n_rows), 3.5, 11.5).round(2)
    inflation_rate = np.clip(rng.normal(5.4, 1.0, n_rows), 2.0, 9.5).round(2)

    interest_rate = (
        8.5
        + debt_to_income * 3.8
        + utilization_rate * 2.2
        + missed_payments_6m * 0.55
        - (credit_score - 650) / 100
        + np.where(secured_loan == "unsecured", 1.25, -0.35)
        + rng.normal(0, 0.7, n_rows)
    )
    interest_rate = np.clip(interest_rate, 6.5, 24.0).round(2)

    business_flag = (loan_purpose == "business").astype(int)
    unsecured_flag = (secured_loan == "unsecured").astype(int)

    log_odds = (
        -4.6
        + 0.015 * (interest_rate - 10)
        + 1.55 * debt_to_income
        + 1.15 * utilization_rate
        + 0.34 * missed_payments_6m
        - 0.0065 * (credit_score - 650)
        - 0.035 * employment_length
        + 0.18 * unsecured_flag
        + 0.22 * business_flag
        + 0.08 * (unemployment_rate - 6.5)
        + 0.05 * (inflation_rate - 5.0)
    )

    probability_default = 1 / (1 + np.exp(-log_odds))
    default_12m = rng.binomial(1, probability_default)

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "employment_length": employment_length,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "debt_to_income": debt_to_income,
            "utilization_rate": utilization_rate,
            "missed_payments_6m": missed_payments_6m,
            "loan_tenure_months": loan_tenure_months,
            "secured_loan": secured_loan,
            "loan_purpose": loan_purpose,
            "region": region,
            "unemployment_rate": unemployment_rate,
            "inflation_rate": inflation_rate,
            "interest_rate": interest_rate,
            "default_12m": default_12m,
        }
    )
