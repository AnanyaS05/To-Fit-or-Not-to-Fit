from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from to_fit_or_not_to_fit.cold_start import (  # noqa: E402
    CLASS_NAMES,
    GARMENT_TYPES,
    SIZE_LABELS,
    Preprocessor,
    build_candidate_rows,
    read_brand_chart,
    transform_features,
)
from to_fit_or_not_to_fit.manual_mlp import ManualMLPClassifier  # noqa: E402
from to_fit_or_not_to_fit.metrics import accuracy_score, macro_f1_score  # noqa: E402


WARNING_THRESHOLD = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cold-start MLP and Bayesian calibration artifacts.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "cold_start")
    parser.add_argument("--brand-csv", type=Path, default=ROOT / "Data" / "demo_brand_sizing.csv")
    parser.add_argument(
        "--calibrated-test-json",
        type=Path,
        default=None,
        help="Optional output JSON from score_brms_calibrator.R for mlp_test_predictions.csv.",
    )
    return parser.parse_args()


def log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        mask = y_true == idx
        recalls[class_name] = float(np.mean(y_pred[mask] == idx)) if mask.any() else float("nan")
    return recalls


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")


def assert_probability_rows_sum_to_one(values: np.ndarray, name: str, tolerance: float = 1e-5) -> float:
    row_sums = values.sum(axis=1)
    max_error = float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) else 0.0
    if max_error > tolerance:
        raise ValueError(f"{name} probabilities do not sum to 1. max_error={max_error:.8f}")
    return max_error


def validate_chart(chart: pd.DataFrame) -> None:
    observed = set(zip(chart["garment_type"], chart["size_label"]))
    expected = {(garment, size) for garment in GARMENT_TYPES for size in SIZE_LABELS}
    missing = expected - observed
    if missing:
        raise ValueError(f"Size chart is missing garment/size rows: {sorted(missing)}")


def validate_training_table(frame: pd.DataFrame) -> None:
    required_cols = {"source", "garment_type", "selected_size_label", "fit_label"}
    missing_cols = required_cols - set(frame.columns)
    if missing_cols:
        raise ValueError(f"Training table is missing columns: {sorted(missing_cols)}")
    if not set(frame["garment_type"]).issubset(set(GARMENT_TYPES)):
        raise ValueError("Training table has unexpected garment types.")
    if not set(frame["selected_size_label"]).issubset(set(SIZE_LABELS)):
        raise ValueError("Training table has unexpected size labels.")
    if not set(frame["fit_label"]).issubset(set(CLASS_NAMES)):
        raise ValueError("Training table has unexpected class labels.")


def validate_mlp_predictions(frame: pd.DataFrame, split_name: str) -> dict[str, float]:
    probabilities = frame[[f"mlp_p_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)
    max_sum_error = assert_probability_rows_sum_to_one(probabilities, f"{split_name} MLP")
    y_true = frame["fit_label"].map({name: idx for idx, name in enumerate(CLASS_NAMES)}).to_numpy(dtype=int)
    y_pred = probabilities.argmax(axis=1)
    return {
        "rows": float(len(frame)),
        "max_probability_sum_error": max_sum_error,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1_score(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))),
        "log_loss": log_loss(y_true, probabilities),
        "brier": multiclass_brier(y_true, probabilities),
        "per_class_recall": per_class_recall(y_true, y_pred),
    }


def validate_calibrated_predictions(
    test_frame: pd.DataFrame,
    calibrated_json: Path,
) -> dict[str, float]:
    payload = json.loads(calibrated_json.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if len(rows) != len(test_frame):
        raise ValueError(
            f"Calibrated row count mismatch: {len(rows)} rows in JSON vs {len(test_frame)} test rows."
        )

    probabilities = np.asarray(
        [[row["probabilities"][name] for name in CLASS_NAMES] for row in rows],
        dtype=float,
    )
    max_sum_error = assert_probability_rows_sum_to_one(probabilities, "calibrated test")
    y_true = test_frame["fit_label"].map({name: idx for idx, name in enumerate(CLASS_NAMES)}).to_numpy(dtype=int)
    y_pred = probabilities.argmax(axis=1)
    misfit_true = test_frame["fit_label"].ne("fit").to_numpy()
    misfit_probabilities = probabilities[:, 0] + probabilities[:, 2]
    warning_mask = misfit_probabilities >= WARNING_THRESHOLD

    warning_precision = (
        float(misfit_true[warning_mask].mean()) if warning_mask.any() else float("nan")
    )
    warning_recall = (
        float(warning_mask[misfit_true].mean()) if misfit_true.any() else float("nan")
    )
    warning_direction = np.where(probabilities[:, 0] >= probabilities[:, 2], "small", "large")
    true_labels = test_frame["fit_label"].to_numpy(dtype=str)
    true_misfit_warning = warning_mask & misfit_true
    warning_direction_accuracy = (
        float(np.mean(warning_direction[true_misfit_warning] == true_labels[true_misfit_warning]))
        if true_misfit_warning.any()
        else float("nan")
    )

    return {
        "rows": float(len(rows)),
        "max_probability_sum_error": max_sum_error,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1_score(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))),
        "log_loss": log_loss(y_true, probabilities),
        "brier": multiclass_brier(y_true, probabilities),
        "per_class_recall": per_class_recall(y_true, y_pred),
        "warning_count_at_0.70": float(warning_mask.sum()),
        "warning_rate_at_0.70": float(warning_mask.mean()),
        "warning_precision_at_0.70": warning_precision,
        "warning_recall_at_0.70": warning_recall,
        "warning_direction_accuracy_at_0.70": warning_direction_accuracy,
        "misfit_probability_min": float(misfit_probabilities.min()),
        "misfit_probability_mean": float(misfit_probabilities.mean()),
        "misfit_probability_max": float(misfit_probabilities.max()),
    }


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    calibrated_json = args.calibrated_test_json or artifact_dir / "calibrated_test_predictions.json"

    model_path = artifact_dir / "manual_mlp_model.npz"
    preprocessor_path = artifact_dir / "preprocessor.json"
    training_table_path = artifact_dir / "cold_start_training_table.csv"
    calibration_path = artifact_dir / "mlp_calibration_predictions.csv"
    test_path = artifact_dir / "mlp_test_predictions.csv"

    for path in [model_path, preprocessor_path, training_table_path, calibration_path, test_path]:
        require_file(path)

    chart = read_brand_chart(args.brand_csv)
    validate_chart(chart)

    training_frame = pd.read_csv(training_table_path)
    validate_training_table(training_frame)

    calibration_frame = pd.read_csv(calibration_path)
    test_frame = pd.read_csv(test_path)
    calibration_metrics = validate_mlp_predictions(calibration_frame, "calibration")
    test_metrics = validate_mlp_predictions(test_frame, "test")

    model = ManualMLPClassifier.load(model_path)
    if model.class_names != CLASS_NAMES:
        raise ValueError(f"Unexpected model class order: {model.class_names}")
    preprocessor = Preprocessor.from_json(preprocessor_path)
    candidates = build_candidate_rows(
        chart,
        garment_type="dresses",
        height_inches=65.0,
        bust=36.0,
        waist=28.0,
        hips=38.0,
        selected_sizes=SIZE_LABELS,
    )
    candidate_matrix = transform_features(candidates, preprocessor)
    candidate_probabilities = model.predict_proba(candidate_matrix)
    candidate_max_sum_error = assert_probability_rows_sum_to_one(
        candidate_probabilities,
        "candidate MLP",
    )

    report = {
        "artifact_dir": str(artifact_dir),
        "training_rows": int(len(training_frame)),
        "calibration_mlp": calibration_metrics,
        "test_mlp": test_metrics,
        "candidate_mlp_rows": int(len(candidates)),
        "candidate_mlp_max_probability_sum_error": candidate_max_sum_error,
    }

    if calibrated_json.exists():
        report["calibrated_test"] = validate_calibrated_predictions(test_frame, calibrated_json)
    else:
        report["calibrated_test"] = "skipped; no calibrated test JSON found"

    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
