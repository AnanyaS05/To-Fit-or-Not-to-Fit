from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from to_fit_or_not_to_fit.cold_start import CLASS_NAMES, FEATURE_COLUMNS  # noqa: E402
from to_fit_or_not_to_fit.metrics import (  # noqa: E402
    accuracy_score,
    classification_report_dataframe,
    confusion_matrix,
    macro_f1_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate cold-start artifacts and compare the MLP, standalone Bayesian, "
            "and optional ensemble predictions."
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "cold_start")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required artifact: {path}. Re-run the cold-start training pipeline "
            "for this artifact directory."
        )


def require_columns(frame: pd.DataFrame, required_columns: list[str], name: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def assert_probability_rows_sum_to_one(values: np.ndarray, name: str, tolerance: float = 1e-5) -> float:
    row_sums = values.sum(axis=1)
    max_error = float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) else 0.0
    if max_error > tolerance:
        raise ValueError(f"{name} probabilities do not sum to 1. max_error={max_error:.8f}")
    return max_error


def log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def validate_split_frame(frame: pd.DataFrame, split_name: str) -> None:
    required_columns = ["row_id", "fit_label", *FEATURE_COLUMNS]
    require_columns(frame, required_columns, f"{split_name} split")

    if frame["row_id"].isna().any():
        raise ValueError(f"{split_name} split has missing row_id values.")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"{split_name} split has duplicate row_id values.")
    if not set(frame["fit_label"]).issubset(set(CLASS_NAMES)):
        raise ValueError(f"{split_name} split has unexpected fit labels.")


def load_split_frame(path: Path, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_split_frame(frame, split_name)
    return frame.copy()


def align_prediction_frame(
    prediction_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    probability_columns: list[str],
    prediction_column: str | None,
    name: str,
) -> pd.DataFrame:
    required_columns = ["row_id", "fit_label", *probability_columns]
    if prediction_column is not None:
        required_columns.append(prediction_column)
    require_columns(prediction_frame, required_columns, name)

    if prediction_frame["row_id"].isna().any():
        raise ValueError(f"{name} has missing row_id values.")
    if prediction_frame["row_id"].duplicated().any():
        raise ValueError(f"{name} has duplicate row_id values.")

    split_ids = set(split_frame["row_id"].tolist())
    prediction_ids = set(prediction_frame["row_id"].tolist())
    if split_ids != prediction_ids:
        missing = sorted(split_ids - prediction_ids)
        extra = sorted(prediction_ids - split_ids)
        raise ValueError(
            f"{name} row_id mismatch. missing={missing[:5]} extra={extra[:5]}"
        )

    split_aligned = split_frame.sort_values("row_id").reset_index(drop=True)
    prediction_aligned = prediction_frame.sort_values("row_id").reset_index(drop=True)

    if not split_aligned["row_id"].equals(prediction_aligned["row_id"]):
        raise ValueError(f"{name} row order does not match the source split after sorting.")

    if not split_aligned["fit_label"].astype(str).equals(prediction_aligned["fit_label"].astype(str)):
        raise ValueError(f"{name} fit_label values do not match the source split.")

    probability_values = prediction_aligned[probability_columns].to_numpy(dtype=float)
    max_probability_sum_error = assert_probability_rows_sum_to_one(probability_values, name)

    if prediction_column is not None:
        predicted_labels = prediction_aligned[prediction_column].astype(str)
        if not set(predicted_labels).issubset(set(CLASS_NAMES)):
            raise ValueError(f"{name} has unexpected predicted classes.")

    output = prediction_aligned.copy()
    output.attrs["max_probability_sum_error"] = max_probability_sum_error
    return output


def build_metrics(frame: pd.DataFrame, probability_columns: list[str], prediction_column: str) -> dict[str, object]:
    labels = np.asarray(CLASS_NAMES, dtype=object)
    y_true_labels = frame["fit_label"].to_numpy(dtype=object)
    y_pred_labels = frame[prediction_column].to_numpy(dtype=object)
    y_true_encoded = frame["fit_label"].map({label: idx for idx, label in enumerate(CLASS_NAMES)}).to_numpy(dtype=int)
    probabilities = frame[probability_columns].to_numpy(dtype=float)

    report_df = classification_report_dataframe(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        label_names=CLASS_NAMES,
    )
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)

    return {
        "rows": int(len(frame)),
        "max_probability_sum_error": float(frame.attrs["max_probability_sum_error"]),
        "accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "macro_f1": float(macro_f1_score(y_true_labels, y_pred_labels, labels=labels)),
        "log_loss": log_loss(y_true_encoded, probabilities),
        "brier": multiclass_brier(y_true_encoded, probabilities),
        "per_class": report_df.to_dict(orient="records"),
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "values": cm.tolist(),
        },
    }


def materialize_argmax_predictions(
    frame: pd.DataFrame,
    probability_columns: list[str],
    prediction_column: str,
) -> pd.DataFrame:
    if prediction_column in frame.columns:
        return frame

    output = frame.copy()
    output[prediction_column] = [
        CLASS_NAMES[index]
        for index in np.argmax(output[probability_columns].to_numpy(dtype=float), axis=1)
    ]
    return output


def compare_scores(scores: dict[str, float]) -> str:
    best_score = max(scores.values())
    winners = [model_name for model_name, score in scores.items() if np.isclose(score, best_score)]
    if len(winners) != 1:
        return "tie"
    return winners[0]


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir

    required_paths = {
        "train_split": artifact_dir / "train_split.csv",
        "calibration_split": artifact_dir / "calibration_split.csv",
        "test_split": artifact_dir / "test_split.csv",
        "mlp_calibration_predictions": artifact_dir / "mlp_calibration_predictions.csv",
        "mlp_test_predictions": artifact_dir / "mlp_test_predictions.csv",
        "bayes_calibration_predictions": artifact_dir / "bayesian_categorical_calibration_predictions.csv",
        "bayes_test_predictions": artifact_dir / "bayesian_categorical_test_predictions.csv",
    }
    optional_ensemble_paths = {
        "ensemble_calibration_predictions": artifact_dir / "ensemble_calibration_predictions.csv",
        "ensemble_test_predictions": artifact_dir / "ensemble_test_predictions.csv",
    }

    for path in required_paths.values():
        require_file(path)

    present_ensemble_paths = [path.exists() for path in optional_ensemble_paths.values()]
    if any(present_ensemble_paths) and not all(present_ensemble_paths):
        missing = [name for name, path in optional_ensemble_paths.items() if not path.exists()]
        raise FileNotFoundError(
            "Incomplete ensemble artifact set. Missing: "
            f"{missing}. Re-run scripts/build_submission_ensemble.py for this artifact directory."
        )
    use_ensemble = all(present_ensemble_paths)
    if use_ensemble:
        required_paths.update(optional_ensemble_paths)

    train_frame = load_split_frame(required_paths["train_split"], "train")
    calibration_frame = load_split_frame(required_paths["calibration_split"], "calibration")
    test_frame = load_split_frame(required_paths["test_split"], "test")

    mlp_calibration_raw = pd.read_csv(required_paths["mlp_calibration_predictions"])
    mlp_prediction_column = "mlp_pred_class" if "mlp_pred_class" in mlp_calibration_raw.columns else None
    mlp_calibration = align_prediction_frame(
        mlp_calibration_raw,
        calibration_frame,
        [f"mlp_p_{class_name}" for class_name in CLASS_NAMES],
        mlp_prediction_column,
        "mlp calibration predictions",
    )
    mlp_calibration = materialize_argmax_predictions(
        mlp_calibration,
        [f"mlp_p_{class_name}" for class_name in CLASS_NAMES],
        "mlp_pred_class",
    )

    mlp_test_raw = pd.read_csv(required_paths["mlp_test_predictions"])
    mlp_prediction_column = "mlp_pred_class" if "mlp_pred_class" in mlp_test_raw.columns else None
    mlp_test = align_prediction_frame(
        mlp_test_raw,
        test_frame,
        [f"mlp_p_{class_name}" for class_name in CLASS_NAMES],
        mlp_prediction_column,
        "mlp test predictions",
    )
    mlp_test = materialize_argmax_predictions(
        mlp_test,
        [f"mlp_p_{class_name}" for class_name in CLASS_NAMES],
        "mlp_pred_class",
    )

    bayes_calibration = align_prediction_frame(
        pd.read_csv(required_paths["bayes_calibration_predictions"]),
        calibration_frame,
        [f"bayes_p_{class_name}" for class_name in CLASS_NAMES],
        "bayes_pred_class",
        "bayesian calibration predictions",
    )
    bayes_test = align_prediction_frame(
        pd.read_csv(required_paths["bayes_test_predictions"]),
        test_frame,
        [f"bayes_p_{class_name}" for class_name in CLASS_NAMES],
        "bayes_pred_class",
        "bayesian test predictions",
    )

    mlp_calibration_metrics = build_metrics(
        mlp_calibration,
        [f"mlp_p_{class_name}" for class_name in CLASS_NAMES],
        "mlp_pred_class",
    )
    mlp_test_metrics = build_metrics(
        mlp_test,
        [f"mlp_p_{class_name}" for class_name in CLASS_NAMES],
        "mlp_pred_class",
    )
    bayes_calibration_metrics = build_metrics(
        bayes_calibration,
        [f"bayes_p_{class_name}" for class_name in CLASS_NAMES],
        "bayes_pred_class",
    )
    bayes_test_metrics = build_metrics(
        bayes_test,
        [f"bayes_p_{class_name}" for class_name in CLASS_NAMES],
        "bayes_pred_class",
    )

    calibration_metrics = {
        "mlp": mlp_calibration_metrics,
        "bayesian": bayes_calibration_metrics,
    }
    test_metrics = {
        "mlp": mlp_test_metrics,
        "bayesian": bayes_test_metrics,
    }

    if use_ensemble:
        ensemble_calibration = align_prediction_frame(
            pd.read_csv(required_paths["ensemble_calibration_predictions"]),
            calibration_frame,
            [f"ensemble_p_{class_name}" for class_name in CLASS_NAMES],
            "ensemble_pred_class",
            "ensemble calibration predictions",
        )
        ensemble_test = align_prediction_frame(
            pd.read_csv(required_paths["ensemble_test_predictions"]),
            test_frame,
            [f"ensemble_p_{class_name}" for class_name in CLASS_NAMES],
            "ensemble_pred_class",
            "ensemble test predictions",
        )
        calibration_metrics["ensemble"] = build_metrics(
            ensemble_calibration,
            [f"ensemble_p_{class_name}" for class_name in CLASS_NAMES],
            "ensemble_pred_class",
        )
        test_metrics["ensemble"] = build_metrics(
            ensemble_test,
            [f"ensemble_p_{class_name}" for class_name in CLASS_NAMES],
            "ensemble_pred_class",
        )

    winner_scores = {model_name: float(metrics["macro_f1"]) for model_name, metrics in test_metrics.items()}
    winner = {
        "metric": "test_macro_f1",
        "model": compare_scores(winner_scores),
        "mlp": winner_scores["mlp"],
        "bayesian": winner_scores["bayesian"],
    }
    if "ensemble" in winner_scores:
        winner["ensemble"] = winner_scores["ensemble"]

    report = {
        "artifact_dir": str(artifact_dir),
        "split_counts": {
            "train": int(len(train_frame)),
            "calibration": int(len(calibration_frame)),
            "test": int(len(test_frame)),
        },
        "calibration": calibration_metrics,
        "test": test_metrics,
        "primary_metric": "test_macro_f1",
        "winner": winner,
        "paths": {name: str(path) for name, path in required_paths.items()},
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
