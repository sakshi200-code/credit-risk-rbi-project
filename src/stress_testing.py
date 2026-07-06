import pandas as pd


def run_stress_tests(model, feature_columns: list[str], data: pd.DataFrame) -> pd.DataFrame:
    baseline = data[feature_columns].copy()
    baseline_pd = model.predict_proba(baseline).mean()

    scenarios = {
        "baseline": {},
        "mild_stress": {
            "loan_int_rate": 0.75,
            "loan_percent_income": 0.03,
        },
        "severe_stress": {
            "loan_int_rate": 1.75,
            "loan_percent_income": 0.07,
        },
    }

    rows = []
    for scenario_name, shocks in scenarios.items():
        scenario_data = baseline.copy()
        for column, shock in shocks.items():
            if column in scenario_data.columns:
                scenario_data[column] = scenario_data[column] + shock

        avg_pd = model.predict_proba(scenario_data).mean()
        rows.append(
            {
                "scenario": scenario_name,
                "average_pd": avg_pd,
                "absolute_change": avg_pd - baseline_pd,
                "relative_change_pct": ((avg_pd / baseline_pd) - 1) * 100,
            }
        )

    return pd.DataFrame(rows)
