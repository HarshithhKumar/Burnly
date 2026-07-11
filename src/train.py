import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    data = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n_samples),
            "gender": rng.choice(["male", "female"], size=n_samples),
            "height_cm": rng.normal(170, 10, size=n_samples).round(1),
            "weight_kg": rng.normal(70, 12, size=n_samples).round(1),
            "duration_min": rng.integers(10, 90, size=n_samples),
            "heart_rate": rng.normal(125, 15, size=n_samples).round(1),
        }
    )

    gender_map = {"male": 1, "female": 0}
    data["gender_code"] = data["gender"].map(gender_map)

    calories = (
        8.0 * data["age"]
        + 12.0 * data["gender_code"]
        + 0.4 * data["height_cm"]
        + 0.6 * data["weight_kg"]
        + 1.8 * data["duration_min"]
        + 0.9 * data["heart_rate"]
        - 350
    )
    data["calories"] = np.maximum(100, np.round(calories + rng.normal(0, 25, size=n_samples), 1))

    return data[["age", "gender", "height_cm", "weight_kg", "duration_min", "heart_rate", "calories"]]


def load_or_generate_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    if data_path is None:
        data_path = Path("data/synthetic_calories.csv")

    data_path = Path(data_path)
    if data_path.exists():
        return pd.read_csv(data_path)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_dataset()
    data.to_csv(data_path, index=False)
    return data


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = data.drop(columns=["calories"])
    features = pd.get_dummies(features, columns=["gender"], dtype=float)
    target = data["calories"]
    return features, target


def train_and_evaluate(data: pd.DataFrame, output_dir: Optional[Path] = None) -> dict:
    if output_dir is None:
        output_dir = Path("artifacts")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=180, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
    }

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Calories")
    plt.ylabel("Predicted Calories")
    plt.title("Actual vs Predicted Calories")
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_plot.png", dpi=200)
    plt.close()

    joblib.dump(model, output_dir / "model.joblib")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def main() -> None:
    data_path = Path("data/synthetic_calories.csv")
    output_dir = Path("artifacts")
    data = load_or_generate_data(data_path)
    metrics = train_and_evaluate(data, output_dir=output_dir)

    print("Training completed successfully.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
