from pathlib import Path

import pandas as pd


TARGET = "loan_status"


def load_credit_risk_dataset(zip_path: Path, output_csv_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(zip_path)
    output_csv_path.parent.mkdir(exist_ok=True)
    raw.to_csv(output_csv_path, index=False)

    cleaned = raw.copy()
    original_rows = len(cleaned)

    cleaned = cleaned[cleaned["person_age"].between(18, 100)]
    cleaned = cleaned[cleaned["person_emp_length"].isna() | cleaned["person_emp_length"].between(0, 60)]
    cleaned = cleaned[cleaned["person_income"] > 0]
    cleaned = cleaned[cleaned["loan_amnt"] > 0]

    audit = {
        "source_rows": original_rows,
        "model_rows": len(cleaned),
        "removed_rows": original_rows - len(cleaned),
        "target_column": TARGET,
        "default_rate": float(cleaned[TARGET].mean()),
        "missing_person_emp_length_pct": float(raw["person_emp_length"].isna().mean() * 100),
        "missing_loan_int_rate_pct": float(raw["loan_int_rate"].isna().mean() * 100),
    }
    return cleaned.reset_index(drop=True), audit
