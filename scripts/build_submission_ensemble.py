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
            "Blend saved MLP and Bayesian prediction artifacts into a calibrated "
            "submission ensemble."
        )
    )
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "submission_final")
    parser.add_argument("--weight-grid-step", type=float, default=0.02)
    parser.add_argument(
        "--decision-mode",
        choices=["argmax", "macro_f1_bias_search"],
        default="macro_f1_bias_search",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")


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


def sort_frame_by_row_id(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("row_id").reset_index(drop=True)


def align_prediction_frame(
    prediction_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    probability_columns: list[str],
    name: str,
) -> pd.DataFrame:
    require_columns(prediction_frame, ["row_id", "fit_label", *probability_columns], name)

    if prediction_frame["row_id"].isna().any():
        raise ValueError(f"{name} has missing row_id values.")
    if prediction_frame["row_id"].duplicated().any():
        raise ValueError(f"{name} has duplicate row_id values.")

    split_ids = set(split_frame["row_id"].tolist())
    prediction_ids = set(prediction_frame["row_id"].tolist())
    if split_ids != prediction_ids:
        missing = sorted(split_ids - prediction_ids)
        extra = sorted(prediction_ids - split_ids)
        raise ValueError(f"{name} row_id mismatch. missing={missing[:5]} extra={extra[:5]}")

    split_aligned = split_frame.sort_values("row_id").reset_index(drop=True)
    prediction_aligned = prediction_frame.sort_values("row_id").reset_index(drop=True)

    if not split_aligned["row_id"].equals(prediction_aligned["row_id"]):
        raise ValueError(f"{name} row order does not match the source split after sorting.")
    if not split_aligned["fit_label"].astype(str).equals(prediction_aligned["fit_label"].astype(str)):
        raise ValueError(f"{name} fit_label values do not match the source split.")

    probability_values = prediction_aligned[probability_columns].to_numpy(dtype=float)
    max_probability_sum_error = assert_probability_rows_sum_to_one(probability_values, name)

    output = prediction_aligned.copy()
    output.attrs["max_probability_sum_error"] = max_probability_sum_error
    return output


def add_probability_columns(frame: pd.DataFrame, probabilities: np.ndarray, prefix: str) -> pd.DataFrame:
    output = frame.copy()
    for idx, class_name in enumerate(CLASS_NAMES):
        clipped = np.clip(probabilities[:, idx], 1e-8, 1.0)
        output[f"{prefix}_p_{class_name}"] = probabilities[:, idx]
        output[f"{prefix}_logit_{class_name}"] = np.log(clipped)
    return output


def predict_argmax_classes(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray([CLASS_NAMES[index] for index in np.argmax(probabilities, axis=1)], dtype=object)


def predict_classes_from_probabilities(
    probabilities: np.ndarray,
    class_biases: dict[str, float],
) -> np.ndarray:
    ordered_biases = np.asarray([float(class_biases[class_name]) for class_name in CLASS_NAMES], dtype=float)
    adjusted_scores = np.log(np.clip(probabilities, 1e-12, 1.0)) + ordered_biases
    return np.asarray([CLASS_NAMES[index] for index in np.argmax(adjusted_scores, axis=1)], dtype=object)


def tune_decision_rule(
    truth_labels: np.ndarray,
    probabilities: np.ndarray,
    grid_min: float = -1.5,
    grid_max: float = 1.5,
    grid_step: float = 0.1,
) -> dict[str, object]:
    default_biases = {class_name: 0.0 for class_name in CLASS_NAMES}
    baseline_predictions = predict_classes_from_probabilities(probabilities, default_biases)
    baseline_macro_f1 = float(macro_f1_score(truth_labels, baseline_predictions, labels=np.asarray(CLASS_NAMES)))

    best_macro_f1 = baseline_macro_f1
    best_biases = default_biases.copy()

    def evaluate_grid(
        small_values: np.ndarray,
        large_values: np.ndarray,
        incumbent_score: float,
        incumbent_biases: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        best_local_score = incumbent_score
        best_local_biases = incumbent_biases.copy()

        for small_bias in small_values:
            for large_bias in large_values:
                candidate_biases = {"small": float(small_bias), "fit": 0.0, "large": float(large_bias)}
                predictions = predict_classes_from_probabilities(probabilities, candidate_biases)
                candidate_score = float(
                    macro_f1_score(truth_labels, predictions, labels=np.asarray(CLASS_NAMES))
                )

                if candidate_score > best_local_score + 1e-12:
                    best_local_score = candidate_score
                    best_local_biases = candidate_biases
                elif np.isclose(candidate_score, best_local_score) and (
                    abs(candidate_biases["small"]) + abs(candidate_biases["large"])
                    < abs(best_local_biases["small"]) + abs(best_local_biases["large"])
                ):
                    best_local_biases = candidate_biases

        return best_local_score, best_local_biases

    coarse_values = np.arange(grid_min, grid_max + 1e-9, grid_step, dtype=float)
    best_macro_f1, best_biases = evaluate_grid(coarse_values, coarse_values, best_macro_f1, best_biases)

    fine_step = grid_step / 5.0
    if fine_step > 0:
        small_center = best_biases["small"]
        large_center = best_biases["large"]
        fine_small = np.arange(
            max(grid_min, small_center - grid_step),
            min(grid_max, small_center + grid_step) + 1e-9,
            fine_step,
            dtype=float,
        )
        fine_large = np.arange(
            max(grid_min, large_center - grid_step),
            min(grid_max, large_center + grid_step) + 1e-9,
            fine_step,
            dtype=float,
        )
        best_macro_f1, best_biases = evaluate_grid(fine_small, fine_large, best_macro_f1, best_biases)

    return {
        "mode": "macro_f1_bias_search",
        "class_biases": [best_biases[class_name] for class_name in CLASS_NAMES],
        "class_biases_by_label": best_biases,
        "calibration_macro_f1": best_macro_f1,
        "baseline_argmax_macro_f1": baseline_macro_f1,
    }


def metrics_from_probabilities(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, object]:
    y_true_labels = frame["fit_label"].to_numpy(dtype=object)
    y_true = frame["fit_label"].map({label: idx for idx, label in enumerate(CLASS_NAMES)}).to_numpy(dtype=int)
    labels = np.asarray(CLASS_NAMES, dtype=object)
    report_df = classification_report_dataframe(
        y_true_labels,
        predicted_labels,
        labels=labels,
        label_names=CLASS_NAMES,
    )
    cm = confusion_matrix(y_true_labels, predicted_labels, labels=labels)
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(y_true_labels, predicted_labels)),
        "macro_f1": float(macro_f1_score(y_true_labels, predicted_labels, labels=labels)),
        "log_loss": log_loss(y_true, probabilities),
        "brier": multiclass_brier(y_true, probabilities),
        "per_class": report_df.to_dict(orient="records"),
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "values": cm.tolist(),
        },
    }


def build_weight_grid(step: float) -> np.ndarray:
    if step <= 0.0 or step > 1.0:
        raise ValueError("--weight-grid-step must be in the interval (0, 1].")

    values = np.arange(0.0, 1.0 + 1e-9, step, dtype=float)
    values = np.unique(np.round(np.clip(values, 0.0, 1.0), 10))
    if not np.isclose(values[0], 0.0):
        values = np.insert(values, 0, 0.0)
    if not np.isclose(values[-1], 1.0):
        values = np.append(values, 1.0)
    return values


def selection_key(metrics: dict[str, object], mlp_weight: float) -> tuple[float, float, float, float]:
    return (
        float(metrics["macro_f1"]),
        -float(metrics["log_loss"]),
        -float(metrics["brier"]),
        -abs(float(mlp_weight) - 0.50),
    )


def build_decision_rule(
    decision_mode: str,
    truth_labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    if decision_mode == "argmax":
        predictions = predict_argmax_classes(probabilities)
        macro_f1 = float(macro_f1_score(truth_labels, predictions, labels=np.asarray(CLASS_NAMES)))
        return (
            {
                "mode": "argmax",
                "class_biases": [0.0 for _ in CLASS_NAMES],
                "class_biases_by_label": {class_name: 0.0 for class_name in CLASS_NAMES},
                "calibration_macro_f1": macro_f1,
                "baseline_argmax_macro_f1": macro_f1,
            },
            predictions,
        )

    decision_rule = tune_decision_rule(truth_labels, probabilities)
    predictions = predict_classes_from_probabilities(probabilities, decision_rule["class_biases_by_label"])
    return decision_rule, predictions


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir

    required_paths = {
        "train_split": artifact_dir / "train_split.csv",
        "calibration_split": artifact_dir / "calibration_split.csv",
        "test_split": artifact_dir / "test_split.csv",
        "mlp_calibration_predictions": artifact_dir / "mlp_calibration_predictions.csv",
        "mlp_test_predictions": artifact_dir / "mlp_test_predictions.csv",
        "bayesian_calibration_predictions": artifact_dir / "bayesian_categorical_calibration_predictions.csv",
        "bayesian_test_predictions": artifact_dir / "bayesian_categorical_test_predictions.csv",
    }

    for path in required_paths.values():
        require_file(path)

    load_split_frame(required_paths["train_split"], "train")
    calibration_frame = sort_frame_by_row_id(load_split_frame(required_paths["calibration_split"], "calibration"))
    test_frame = sort_frame_by_row_id(load_split_frame(required_paths["test_split"], "test"))

    mlp_calibration = align_prediction_frame(
        pd.read_csv(required_paths["mlp_calibration_predictions"]),
        calibration_frame,
        [f"mlp_p_{class_name}" for class_name in CLASS_NAMES],
        "mlp calibration predictions",
    )
    mlp_test = align_prediction_frame(
        pd.read_csv(required_paths["mlp_test_predictions"]),
        test_frame,
        [f"mlp_p_{class_name}" for class_name in CLASS_NAMES],
        "mlp test predictions",
    )
    bayes_calibration = align_prediction_frame(
        pd.read_csv(required_paths["bayesian_calibration_predictions"]),
        calibration_frame,
        [f"bayes_p_{class_name}" for class_name in CLASS_NAMES],
        "bayesian calibration predictions",
    )
    bayes_test = align_prediction_frame(
        pd.read_csv(required_paths["bayesian_test_predictions"]),
        test_frame,
        [f"bayes_p_{class_name}" for class_name in CLASS_NAMES],
        "bayesian test predictions",
    )

    mlp_calibration_probabilities = mlp_calibration[[f"mlp_p_{class_name}" for class_name in CLASS_NAMES]].to_numpy(
        dtype=float
    )
    mlp_test_probabilities = mlp_test[[f"mlp_p_{class_name}" for class_name in CLASS_NAMES]].to_numpy(dtype=float)
    bayes_calibration_probabilities = bayes_calibration[
        [f"bayes_p_{class_name}" for class_name in CLASS_NAMES]
    ].to_numpy(dtype=float)
    bayes_test_probabilities = bayes_test[[f"bayes_p_{class_name}" for class_name in CLASS_NAMES]].to_numpy(
        dtype=float
    )

    calibration_truth = calibration_frame["fit_label"].to_numpy(dtype=object)
    candidate_records: list[dict[str, object]] = []
    best_candidate: dict[str, object] | None = None

    for mlp_weight in build_weight_grid(args.weight_grid_step):
        bayesian_weight = float(1.0 - mlp_weight)
        calibration_probabilities = (
            float(mlp_weight) * mlp_calibration_probabilities + bayesian_weight * bayes_calibration_probabilities
        )
        decision_rule, calibration_predictions = build_decision_rule(
            args.decision_mode,
            calibration_truth,
            calibration_probabilities,
        )
        calibration_metrics = metrics_from_probabilities(
            calibration_frame,
            calibration_probabilities,
            calibration_predictions,
        )
        record = {
            "mlp_weight": float(mlp_weight),
            "bayesian_weight": bayesian_weight,
            "decision_rule": decision_rule,
            "calibration_metrics": calibration_metrics,
        }
        candidate_records.append(record)

        if best_candidate is None or selection_key(calibration_metrics, mlp_weight) > selection_key(
            best_candidate["calibration_metrics"],
            best_candidate["mlp_weight"],
        ):
            best_candidate = record

    assert best_candidate is not None

    selected_mlp_weight = float(best_candidate["mlp_weight"])
    selected_bayes_weight = float(best_candidate["bayesian_weight"])
    selected_decision_rule = best_candidate["decision_rule"]
    selected_biases = selected_decision_rule["class_biases_by_label"]

    ensemble_calibration_probabilities = (
        selected_mlp_weight * mlp_calibration_probabilities + selected_bayes_weight * bayes_calibration_probabilities
    )
    ensemble_test_probabilities = (
        selected_mlp_weight * mlp_test_probabilities + selected_bayes_weight * bayes_test_probabilities
    )
    ensemble_calibration_predictions = predict_classes_from_probabilities(
        ensemble_calibration_probabilities,
        selected_biases,
    )
    ensemble_test_predictions = predict_classes_from_probabilities(ensemble_test_probabilities, selected_biases)

    ensemble_calibration_export = add_probability_columns(
        calibration_frame,
        ensemble_calibration_probabilities,
        "ensemble",
    )
    ensemble_calibration_export["ensemble_pred_class"] = ensemble_calibration_predictions
    ensemble_calibration_export.to_csv(artifact_dir / "ensemble_calibration_predictions.csv", index=False)

    ensemble_test_export = add_probability_columns(test_frame, ensemble_test_probabilities, "ensemble")
    ensemble_test_export["ensemble_pred_class"] = ensemble_test_predictions
    ensemble_test_export.to_csv(artifact_dir / "ensemble_test_predictions.csv", index=False)

    metadata = {
        "status": "built",
        "artifact_dir": str(artifact_dir),
        "class_names": CLASS_NAMES,
        "decision_mode": args.decision_mode,
        "mlp_weight": selected_mlp_weight,
        "bayesian_weight": selected_bayes_weight,
        "decision_rule": selected_decision_rule,
        "selection": {
            "objective": "calibration_macro_f1",
            "tie_breakers": [
                "lower_log_loss",
                "lower_brier",
                "mlp_weight_closest_to_0.50",
            ],
            "weight_grid_step": float(args.weight_grid_step),
            "candidate_count": len(candidate_records),
        },
        "calibration_metrics": metrics_from_probabilities(
            calibration_frame,
            ensemble_calibration_probabilities,
            ensemble_calibration_predictions,
        ),
        "test_metrics": metrics_from_probabilities(
            test_frame,
            ensemble_test_probabilities,
            ensemble_test_predictions,
        ),
        "paths": {
            "train_split": str(required_paths["train_split"]),
            "calibration_split": str(required_paths["calibration_split"]),
            "test_split": str(required_paths["test_split"]),
            "mlp_calibration_predictions": str(required_paths["mlp_calibration_predictions"]),
            "mlp_test_predictions": str(required_paths["mlp_test_predictions"]),
            "bayesian_calibration_predictions": str(required_paths["bayesian_calibration_predictions"]),
            "bayesian_test_predictions": str(required_paths["bayesian_test_predictions"]),
            "ensemble_calibration_predictions": str(artifact_dir / "ensemble_calibration_predictions.csv"),
            "ensemble_test_predictions": str(artifact_dir / "ensemble_test_predictions.csv"),
            "metadata": str(artifact_dir / "ensemble_metadata.json"),
        },
    }
    save_json(artifact_dir / "ensemble_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
