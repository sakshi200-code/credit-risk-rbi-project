from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CreditRiskModel:
    coefficients: np.ndarray
    feature_names: list[str]
    numeric_columns: list[str]
    means: pd.Series
    stds: pd.Series
    dummy_columns: list[str]

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        numeric = X[self.numeric_columns].fillna(self.means)
        numeric = (numeric - self.means) / self.stds.replace(0, 1)

        categorical = X.drop(columns=self.numeric_columns)
        dummies = pd.get_dummies(categorical, drop_first=False, dtype=float)
        dummies = dummies.reindex(columns=self.dummy_columns, fill_value=0.0)

        prepared = pd.concat([numeric.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        prepared.insert(0, "intercept", 1.0)
        return prepared[self.feature_names]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        prepared = self._prepare(X)
        logits = prepared.to_numpy() @ self.coefficients
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


def _train_test_split(data: pd.DataFrame, target: str, test_size: float = 0.25, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []

    for _, group in data.groupby(target):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        test_count = int(round(len(indices) * test_size))
        test_parts.append(data.loc[indices[:test_count]])
        train_parts.append(data.loc[indices[test_count:]])

    train = pd.concat(train_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train, test


def _prepare_training_matrix(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series, list[str]]:
    numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric = X[numeric_columns].copy()
    means = numeric.mean()
    stds = numeric.std().replace(0, 1)
    numeric = numeric.fillna(means)
    numeric = (numeric - means) / stds

    categorical = X.drop(columns=numeric_columns)
    dummies = pd.get_dummies(categorical, drop_first=False, dtype=float)
    dummy_columns = dummies.columns.tolist()

    prepared = pd.concat([numeric.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    prepared.insert(0, "intercept", 1.0)
    return prepared, numeric_columns, means, stds, dummy_columns


def _fit_logistic_regression(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.08, epochs: int = 2500) -> np.ndarray:
    coefficients = np.zeros(X.shape[1])

    for _ in range(epochs):
        logits = X @ coefficients
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        gradient = X.T @ (probabilities - y) / len(y)
        gradient[1:] += 0.001 * coefficients[1:]
        coefficients -= learning_rate * gradient

    return coefficients


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    positive_count = y_true.sum()
    negative_count = len(y_true) - positive_count
    positive_rank_sum = ranks[y_true == 1].sum()
    return float((positive_rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    frame = pd.DataFrame({"actual": y_true, "score": y_score}).sort_values("score", ascending=False)
    total_bad = frame["actual"].sum()
    total_good = len(frame) - total_bad
    frame["cum_bad"] = frame["actual"].cumsum() / total_bad
    frame["cum_good"] = (1 - frame["actual"]).cumsum() / total_good
    return float((frame["cum_bad"] - frame["cum_good"]).abs().max())


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / len(y_true)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _risk_bands(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": y_true, "pd": y_score})
    frame["risk_band"] = pd.qcut(
        frame["pd"],
        q=5,
        labels=["A_lowest", "B_low", "C_medium", "D_high", "E_highest"],
        duplicates="drop",
    )
    return (
        frame.groupby("risk_band", observed=False)
        .agg(accounts=("actual", "size"), avg_pd=("pd", "mean"), default_rate=("actual", "mean"))
        .reset_index()
    )


def _top_coefficients(model: CreditRiskModel, top_n: int = 12) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": model.feature_names,
            "coefficient": model.coefficients,
        }
    )
    frame = frame[frame["feature"] != "intercept"].copy()
    frame["absolute_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("absolute_coefficient", ascending=False).head(top_n)


def train_and_evaluate(data: pd.DataFrame, target: str) -> dict:
    feature_columns = [column for column in data.columns if column != target]
    train, test = _train_test_split(data, target)

    X_train = train[feature_columns]
    y_train = train[target].to_numpy()
    X_test = test[feature_columns]
    y_test = test[target].to_numpy()

    prepared_train, numeric_columns, means, stds, dummy_columns = _prepare_training_matrix(X_train)
    coefficients = _fit_logistic_regression(prepared_train.to_numpy(), y_train)

    model = CreditRiskModel(
        coefficients=coefficients,
        feature_names=prepared_train.columns.tolist(),
        numeric_columns=numeric_columns,
        means=means,
        stds=stds,
        dummy_columns=dummy_columns,
    )

    predicted_pd = model.predict_proba(X_test)
    threshold = 0.30
    predicted_class = (predicted_pd >= threshold).astype(int)

    metrics = {
        "default_rate": float(data[target].mean()),
        "classification_threshold": threshold,
        "auc": _auc_score(y_test, predicted_pd),
        "ks": _ks_statistic(y_test, predicted_pd),
        **_classification_metrics(y_test, predicted_class),
    }

    return {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "risk_bands": _risk_bands(y_test, predicted_pd),
        "top_coefficients": _top_coefficients(model),
    }
