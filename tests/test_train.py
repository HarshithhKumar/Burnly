from pathlib import Path

from src.train import load_or_generate_data, train_and_evaluate


def test_load_or_generate_data_creates_dataset(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.csv"
    data = load_or_generate_data(data_path)

    assert not data.empty
    assert {"age", "gender", "height_cm", "weight_kg", "duration_min", "heart_rate", "calories"}.issubset(data.columns)


def test_train_and_evaluate_writes_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.csv"
    output_dir = tmp_path / "artifacts"
    data = load_or_generate_data(data_path)
    metrics = train_and_evaluate(data, output_dir=output_dir)

    assert metrics["r2_score"] >= 0.0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "evaluation_plot.png").exists()
