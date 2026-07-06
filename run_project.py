from pathlib import Path

from src.data_loading import TARGET, load_credit_risk_dataset
from src.model_pipeline import train_and_evaluate
from src.reporting import write_interview_notes, write_model_summary
from src.stress_testing import run_stress_tests


BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "data" / "archive.zip"
DATA_PATH = BASE_DIR / "data" / "credit_risk_dataset.csv"
REPORT_PATH = BASE_DIR / "reports" / "model_summary.md"
NOTES_PATH = BASE_DIR / "reports" / "interview_notes.md"


def main() -> None:
    BASE_DIR.joinpath("data").mkdir(exist_ok=True)
    BASE_DIR.joinpath("reports").mkdir(exist_ok=True)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset zip at {ZIP_PATH}. Place archive.zip in the data folder and rerun."
        )

    data, audit = load_credit_risk_dataset(ZIP_PATH, DATA_PATH)
    results = train_and_evaluate(data, target=TARGET)
    stress_results = run_stress_tests(results["model"], results["feature_columns"], data)

    write_model_summary(REPORT_PATH, results, stress_results, audit)
    write_interview_notes(NOTES_PATH)

    print(f"Dataset written to: {DATA_PATH}")
    print(f"Model summary written to: {REPORT_PATH}")
    print(f"Interview notes written to: {NOTES_PATH}")


if __name__ == "__main__":
    main()
