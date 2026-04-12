from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from to_fit_or_not_to_fit.cold_start import (  # noqa: E402
    CLASS_NAMES,
    Preprocessor,
    encode_labels,
    transform_features,
)
from to_fit_or_not_to_fit.manual_mlp import ManualMLPClassifier, ManualMLPConfig  # noqa: E402
from to_fit_or_not_to_fit.metrics import (  # noqa: E402
    accuracy_score,
    classification_report_dataframe,
    confusion_matrix,
    macro_f1_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune and train submission-ready cold-start MLP and Bayesian categorical "
            "models, then export a side-by-side comparison report."
        )
    )
    parser.add_argument("--modcloth-json", type=Path, default=ROOT / "Data" / "modcloth_final_data.json")
    parser.add_argument("--rtr-json", type=Path, default=ROOT / "Data" / "renttherunway_final_data.json")
    parser.add_argument("--brand-csv", type=Path, default=ROOT / "Data" / "demo_brand_sizing.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "submission_final")
    parser.add_argument("--sample-frac-modcloth", type=float, default=1.0)
    parser.add_argument("--sample-frac-rtr", type=float, default=1.0)
    parser.add_argument("--calib-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--profile", choices=["quick", "submission"], default="submission")
    parser.add_argument("--models", choices=["both", "bayesian"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-splits", action="store_true")
    parser.add_argument(
        "--bayesian-decision-mode",
        choices=["argmax", "macro_f1_bias_search"],
        default="macro_f1_bias_search",
    )
    parser.add_argument("--rscript", type=Path, default=None)
    return parser.parse_args()


def find_rscript(explicit: Path | None) -> str:
    if explicit is not None:
        return str(explicit)
    discovered = shutil.which("Rscript")
    if discovered is not None:
        return discovered
    known = Path("C:/Program Files/R/R-4.5.2/bin/Rscript.exe")
    if known.exists():
        return str(known)
    return "Rscript"


def run_command(command: list[str], cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    return completed.stdout


def log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def build_class_weights(y_train: np.ndarray, num_classes: int, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None

    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    safe_counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * safe_counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    return (weights / np.mean(weights)).astype(np.float32)


def load_split_frames(artifact_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(artifact_dir / "train_split.csv")
    calib_df = pd.read_csv(artifact_dir / "calibration_split.csv")
    test_df = pd.read_csv(artifact_dir / "test_split.csv")
    return train_df, calib_df, test_df


def prepare_split_matrices(
    artifact_dir: Path,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Preprocessor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = Preprocessor.from_json(artifact_dir / "preprocessor.json")
    X_train = transform_features(train_df, preprocessor)
    X_calib = transform_features(calib_df, preprocessor)
    X_test = transform_features(test_df, preprocessor)
    y_train = encode_labels(train_df["fit_label"])
    y_calib = encode_labels(calib_df["fit_label"])
    y_test = encode_labels(test_df["fit_label"])
    return preprocessor, X_train, X_calib, X_test, y_train, y_calib, y_test


def add_probability_columns(frame: pd.DataFrame, probabilities: np.ndarray, prefix: str) -> pd.DataFrame:
    output = frame.copy()
    for idx, class_name in enumerate(CLASS_NAMES):
        clipped = np.clip(probabilities[:, idx], 1e-8, 1.0)
        output[f"{prefix}_p_{class_name}"] = probabilities[:, idx]
        output[f"{prefix}_logit_{class_name}"] = np.log(clipped)
    return output


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


def selection_key(metrics: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(metrics["macro_f1"]),
        -float(metrics["log_loss"]),
        -float(metrics["brier"]),
    )


def predict_classes_from_probabilities(
    probabilities: np.ndarray,
    class_biases: dict[str, float],
) -> np.ndarray:
    ordered_biases = np.asarray([float(class_biases[class_name]) for class_name in CLASS_NAMES], dtype=float)
    adjusted_scores = np.log(np.clip(probabilities, 1e-12, 1.0)) + ordered_biases
    return np.asarray([CLASS_NAMES[idx] for idx in np.argmax(adjusted_scores, axis=1)], dtype=object)


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


def mlp_candidates(profile: str) -> list[dict[str, object]]:
    if profile == "quick":
        return [
            {
                "name": "mlp_trial_01",
                "hidden_dims": [256, 128],
                "learning_rate": 1e-3,
                "batch_size": 512,
                "max_epochs": 14,
                "patience": 5,
                "dropout": 0.15,
                "weight_decay": 1e-4,
                "class_weighting": "balanced",
            },
            {
                "name": "mlp_trial_02",
                "hidden_dims": [256, 128],
                "learning_rate": 8e-4,
                "batch_size": 512,
                "max_epochs": 16,
                "patience": 5,
                "dropout": 0.10,
                "weight_decay": 5e-5,
                "class_weighting": "balanced",
            },
            {
                "name": "mlp_trial_03",
                "hidden_dims": [320, 160],
                "learning_rate": 8e-4,
                "batch_size": 512,
                "max_epochs": 16,
                "patience": 5,
                "dropout": 0.12,
                "weight_decay": 1e-4,
                "class_weighting": "balanced",
            },
            {
                "name": "mlp_trial_04",
                "hidden_dims": [256, 128, 64],
                "learning_rate": 1e-3,
                "batch_size": 512,
                "max_epochs": 16,
                "patience": 5,
                "dropout": 0.12,
                "weight_decay": 5e-5,
                "class_weighting": "balanced",
            },
            {
                "name": "mlp_trial_05",
                "hidden_dims": [192, 96],
                "learning_rate": 1.2e-3,
                "batch_size": 512,
                "max_epochs": 14,
                "patience": 5,
                "dropout": 0.10,
                "weight_decay": 5e-5,
                "class_weighting": "balanced",
            },
        ]

    return [
        {
            "name": "mlp_trial_01",
            "hidden_dims": [256, 128],
            "learning_rate": 1e-3,
            "batch_size": 512,
            "max_epochs": 30,
            "patience": 6,
            "dropout": 0.15,
            "weight_decay": 1e-4,
            "class_weighting": "balanced",
        },
        {
            "name": "mlp_trial_02",
            "hidden_dims": [256, 128],
            "learning_rate": 7.5e-4,
            "batch_size": 512,
            "max_epochs": 35,
            "patience": 7,
            "dropout": 0.20,
            "weight_decay": 3e-4,
            "class_weighting": "sqrt_balanced",
        },
        {
            "name": "mlp_trial_03",
            "hidden_dims": [384, 192],
            "learning_rate": 7.5e-4,
            "batch_size": 512,
            "max_epochs": 35,
            "patience": 7,
            "dropout": 0.15,
            "weight_decay": 1e-4,
            "class_weighting": "sqrt_balanced",
        },
        {
            "name": "mlp_trial_04",
            "hidden_dims": [256, 128, 64],
            "learning_rate": 8e-4,
            "batch_size": 256,
            "max_epochs": 40,
            "patience": 8,
            "dropout": 0.20,
            "weight_decay": 1e-4,
            "class_weighting": "sqrt_balanced",
        },
        {
            "name": "mlp_trial_05",
            "hidden_dims": [192, 96],
            "learning_rate": 1.2e-3,
            "batch_size": 512,
            "max_epochs": 30,
            "patience": 6,
            "dropout": 0.10,
            "weight_decay": 5e-5,
            "class_weighting": "balanced",
        },
        {
            "name": "mlp_trial_06",
            "hidden_dims": [320, 160],
            "learning_rate": 6e-4,
            "batch_size": 256,
            "max_epochs": 40,
            "patience": 8,
            "dropout": 0.25,
            "weight_decay": 2e-4,
            "class_weighting": "sqrt_balanced",
        },
    ]


def bayesian_candidates(profile: str) -> list[dict[str, object]]:
    if profile == "quick":
        return [
            {
                "name": "bayes_trial_01",
                "formula_preset": "compact",
                "weighting": "none",
                "prior_scale": 1.0,
                "intercept_prior_scale": 1.0,
            },
            {
                "name": "bayes_trial_02",
                "formula_preset": "full",
                "weighting": "balanced",
                "prior_scale": 1.0,
                "intercept_prior_scale": 1.5,
            },
            {
                "name": "bayes_trial_03",
                "formula_preset": "full",
                "weighting": "none",
                "prior_scale": 0.75,
                "intercept_prior_scale": 1.0,
            },
        ]

    return [
        {
            "name": "bayes_trial_01",
            "formula_preset": "compact",
            "weighting": "none",
            "prior_scale": 1.0,
            "intercept_prior_scale": 1.0,
        },
        {
            "name": "bayes_trial_02",
            "formula_preset": "full",
            "weighting": "none",
            "prior_scale": 1.0,
            "intercept_prior_scale": 1.0,
        },
        {
            "name": "bayes_trial_03",
            "formula_preset": "compact",
            "weighting": "balanced",
            "prior_scale": 1.0,
            "intercept_prior_scale": 1.5,
        },
        {
            "name": "bayes_trial_04",
            "formula_preset": "full",
            "weighting": "balanced",
            "prior_scale": 1.0,
            "intercept_prior_scale": 1.5,
        },
        {
            "name": "bayes_trial_05",
            "formula_preset": "compact",
            "weighting": "sqrt_balanced",
            "prior_scale": 0.75,
            "intercept_prior_scale": 1.25,
        },
    ]


def bayesian_runtime(profile: str) -> dict[str, dict[str, object]]:
    if profile == "quick":
        return {
            "search": {
                "algorithm": "meanfield",
                "chains": 1,
                "iter": 500,
                "warmup": 0,
                "adapt_delta": 0.95,
                "max_treedepth": 10,
            },
            "final": {
                "algorithm": "sampling",
                "chains": 1,
                "iter": 500,
                "warmup": 250,
                "adapt_delta": 0.98,
                "max_treedepth": 12,
            },
        }

    return {
        "search": {
            "algorithm": "meanfield",
            "chains": 1,
            "iter": 500,
            "warmup": 0,
            "adapt_delta": 0.97,
            "max_treedepth": 12,
        },
        "final": {
            "algorithm": "sampling",
            "chains": 2,
            "iter": 700,
            "warmup": 350,
            "adapt_delta": 0.995,
            "max_treedepth": 13,
        },
    }


def export_splits(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_cold_start_mlp.py"),
        "--modcloth-json",
        str(args.modcloth_json),
        "--rtr-json",
        str(args.rtr_json),
        "--brand-csv",
        str(args.brand_csv),
        "--output-dir",
        str(args.artifact_dir),
        "--sample-frac-modcloth",
        str(args.sample_frac_modcloth),
        "--sample-frac-rtr",
        str(args.sample_frac_rtr),
        "--calib-size",
        str(args.calib_size),
        "--test-size",
        str(args.test_size),
        "--seed",
        str(args.seed),
        "--export-only",
    ]
    run_command(command)


def run_mlp_trial(
    trial: dict[str, object],
    X_train: np.ndarray,
    X_calib: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_calib: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    config = ManualMLPConfig(
        hidden_dims=tuple(trial["hidden_dims"]),
        learning_rate=float(trial["learning_rate"]),
        batch_size=int(trial["batch_size"]),
        max_epochs=int(trial["max_epochs"]),
        patience=int(trial["patience"]),
        dropout=float(trial["dropout"]),
        weight_decay=float(trial["weight_decay"]),
        seed=seed,
        verbose=False,
    )
    class_weights = build_class_weights(y_train, len(CLASS_NAMES), str(trial["class_weighting"]))
    model = ManualMLPClassifier(
        input_dim=X_train.shape[1],
        num_classes=len(CLASS_NAMES),
        config=config,
        class_weights=class_weights,
        class_names=CLASS_NAMES,
    )
    history = model.fit(X_train, y_train, X_calib, y_calib)

    calib_probabilities = model.predict_proba(X_calib)
    test_probabilities = model.predict_proba(X_test)
    y_calib_labels = np.asarray([CLASS_NAMES[idx] for idx in y_calib], dtype=object)
    y_test_labels = np.asarray([CLASS_NAMES[idx] for idx in y_test], dtype=object)
    decision_rule = tune_decision_rule(y_calib_labels, calib_probabilities)
    class_biases = decision_rule["class_biases_by_label"]
    calib_predictions = predict_classes_from_probabilities(calib_probabilities, class_biases)
    test_predictions = predict_classes_from_probabilities(test_probabilities, class_biases)

    return {
        "trial": trial,
        "history": history,
        "model": model,
        "decision_rule": decision_rule,
        "calibration_metrics": metrics_from_probabilities(
            pd.DataFrame({"fit_label": y_calib_labels}),
            calib_probabilities,
            calib_predictions,
        ),
        "test_metrics": metrics_from_probabilities(
            pd.DataFrame({"fit_label": y_test_labels}),
            test_probabilities,
            test_predictions,
        ),
        "calibration_probabilities": calib_probabilities,
        "calibration_predictions": calib_predictions,
        "test_probabilities": test_probabilities,
        "test_predictions": test_predictions,
    }


def write_mlp_final_artifacts(
    artifact_dir: Path,
    preprocessor: Preprocessor,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_result: dict[str, object],
) -> None:
    model: ManualMLPClassifier = best_result["model"]
    model.save(artifact_dir / "manual_mlp_model.npz")
    preprocessor.to_json(artifact_dir / "preprocessor.json")

    calib_export = add_probability_columns(calib_df, best_result["calibration_probabilities"], "mlp")
    test_export = add_probability_columns(test_df, best_result["test_probabilities"], "mlp")
    calib_export["mlp_pred_class"] = best_result["calibration_predictions"]
    test_export["mlp_pred_class"] = best_result["test_predictions"]
    calib_export.to_csv(artifact_dir / "mlp_calibration_predictions.csv", index=False)
    test_export.to_csv(artifact_dir / "mlp_test_predictions.csv", index=False)

    metadata = {
        "class_names": CLASS_NAMES,
        "feature_names": preprocessor.feature_names,
        "feature_columns": list(preprocessor.numeric_cols) + list(preprocessor.categorical_cols),
        "best_trial": best_result["trial"],
        "decision_rule": best_result["decision_rule"],
        "history": best_result["history"],
        "calibration_metrics": best_result["calibration_metrics"],
        "test_metrics": best_result["test_metrics"],
        "paths": {
            "model": str(artifact_dir / "manual_mlp_model.npz"),
            "preprocessor": str(artifact_dir / "preprocessor.json"),
            "train_split": str(artifact_dir / "train_split.csv"),
            "calibration_split": str(artifact_dir / "calibration_split.csv"),
            "test_split": str(artifact_dir / "test_split.csv"),
            "calibration_predictions": str(artifact_dir / "mlp_calibration_predictions.csv"),
            "test_predictions": str(artifact_dir / "mlp_test_predictions.csv"),
            "size_mapping": str(artifact_dir / "size_label_mapping.csv"),
        },
    }
    (artifact_dir / "mlp_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_bayesian_trial(
    rscript: str,
    artifact_dir: Path,
    output_dir: Path,
    trial: dict[str, object],
    runtime: dict[str, object],
    seed: int,
    decision_mode: str,
) -> dict[str, object]:
    command = [
        rscript,
        str(ROOT / "scripts" / "train_bayesian_categorical.R"),
        "--artifact-dir",
        str(artifact_dir),
        "--output-dir",
        str(output_dir),
        "--chains",
        str(runtime["chains"]),
        "--iter",
        str(runtime["iter"]),
        "--warmup",
        str(runtime["warmup"]),
        "--algorithm",
        str(runtime["algorithm"]),
        "--seed",
        str(seed),
        "--weighting",
        str(trial["weighting"]),
        "--formula-preset",
        str(trial["formula_preset"]),
        "--decision-mode",
        str(decision_mode),
        "--prior-scale",
        str(trial["prior_scale"]),
        "--intercept-prior-scale",
        str(trial["intercept_prior_scale"]),
        "--adapt-delta",
        str(runtime["adapt_delta"]),
        "--max-treedepth",
        str(runtime["max_treedepth"]),
    ]
    run_command(command)
    metrics = json.loads((output_dir / "bayesian_categorical_metrics.json").read_text(encoding="utf-8"))
    return {
        "trial": trial,
        "runtime": runtime,
        "metrics": metrics,
        "output_dir": str(output_dir),
    }


def maybe_export_splits(args: argparse.Namespace) -> None:
    required = [
        args.artifact_dir / "train_split.csv",
        args.artifact_dir / "calibration_split.csv",
        args.artifact_dir / "test_split.csv",
        args.artifact_dir / "preprocessor.json",
    ]
    if all(path.exists() for path in required) and (args.reuse_splits or args.models == "bayesian"):
        return
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    export_splits(args)


def load_existing_mlp_summary(artifact_dir: Path) -> dict[str, object]:
    metadata_path = artifact_dir / "mlp_metadata.json"
    calibration_predictions = artifact_dir / "mlp_calibration_predictions.csv"
    test_predictions = artifact_dir / "mlp_test_predictions.csv"

    missing = [path.name for path in [metadata_path, calibration_predictions, test_predictions] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Bayesian-only submission mode requires existing MLP artifacts in the same artifact "
            f"directory. Missing: {missing}. Run the full submission pipeline once or train the "
            "MLP first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "trial": metadata["best_trial"],
        "decision_rule": metadata.get("decision_rule"),
        "calibration_metrics": metadata["calibration_metrics"],
        "test_metrics": metadata["test_metrics"],
    }


def save_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def maybe_build_ensemble(artifact_dir: Path) -> dict[str, object]:
    required_paths = [
        artifact_dir / "mlp_calibration_predictions.csv",
        artifact_dir / "mlp_test_predictions.csv",
        artifact_dir / "bayesian_categorical_calibration_predictions.csv",
        artifact_dir / "bayesian_categorical_test_predictions.csv",
    ]
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        print(
            "Skipping ensemble build because the required base prediction artifacts "
            f"are missing: {missing}"
        )
        return {
            "status": "skipped",
            "reason": "missing_base_prediction_artifacts",
            "missing": missing,
        }

    print("Building submission ensemble from saved MLP and Bayesian predictions")
    ensemble_output = run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_ensemble.py"),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    return json.loads(ensemble_output)


def select_bayesian_candidate(
    search_result: dict[str, object],
    final_result: dict[str, object],
) -> dict[str, object]:
    search_metrics = search_result["metrics"]["calibration"]
    final_metrics = final_result["metrics"]["calibration"]
    if selection_key(search_metrics) >= selection_key(final_metrics):
        return {
            "source": "search",
            "trial": search_result["trial"],
            "runtime": search_result["runtime"],
            "metrics": search_result["metrics"],
            "output_dir": search_result["output_dir"],
        }

    return {
        "source": "final",
        "trial": final_result["trial"],
        "runtime": final_result["runtime"],
        "metrics": final_result["metrics"],
        "output_dir": final_result["output_dir"],
    }


def promote_bayesian_search_artifacts(search_result: dict[str, object], artifact_dir: Path) -> dict[str, object]:
    source_dir = Path(str(search_result["output_dir"]))
    file_names = [
        "bayesian_categorical_model.rds",
        "bayesian_categorical_summary.txt",
        "bayesian_categorical_calibration_predictions.csv",
        "bayesian_categorical_test_predictions.csv",
    ]
    for file_name in file_names:
        shutil.copy2(source_dir / file_name, artifact_dir / file_name)

    metrics = copy.deepcopy(search_result["metrics"])
    metrics["artifact_dir"] = str(artifact_dir)
    metrics["output_dir"] = str(artifact_dir)
    metrics["paths"] = {
        "model": str(artifact_dir / "bayesian_categorical_model.rds"),
        "summary": str(artifact_dir / "bayesian_categorical_summary.txt"),
        "calibration_predictions": str(artifact_dir / "bayesian_categorical_calibration_predictions.csv"),
        "test_predictions": str(artifact_dir / "bayesian_categorical_test_predictions.csv"),
    }
    save_json(artifact_dir / "bayesian_categorical_metrics.json", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing split artifacts in {args.artifact_dir} with profile={args.profile}")
    maybe_export_splits(args)

    train_df, calib_df, test_df = load_split_frames(args.artifact_dir)
    if args.models == "both":
        preprocessor, X_train, X_calib, X_test, y_train, y_calib, y_test = prepare_split_matrices(
            args.artifact_dir,
            train_df,
            calib_df,
            test_df,
        )

        mlp_results: list[dict[str, object]] = []
        best_mlp_result: dict[str, object] | None = None
        for trial in mlp_candidates(args.profile):
            print(f"Running MLP tuning trial {trial['name']}")
            result = run_mlp_trial(trial, X_train, X_calib, X_test, y_train, y_calib, y_test, args.seed)
            mlp_results.append(
                {
                    "trial": trial,
                    "decision_rule": result["decision_rule"],
                    "history": result["history"],
                    "calibration_metrics": result["calibration_metrics"],
                    "test_metrics": result["test_metrics"],
                }
            )
            if best_mlp_result is None or selection_key(result["calibration_metrics"]) > selection_key(
                best_mlp_result["calibration_metrics"]
            ):
                best_mlp_result = result

        assert best_mlp_result is not None
        print(f"Selected best MLP trial: {best_mlp_result['trial']['name']}")
        write_mlp_final_artifacts(args.artifact_dir, preprocessor, train_df, calib_df, test_df, best_mlp_result)
        save_json(args.artifact_dir / "mlp_tuning_results.json", mlp_results)
        mlp_summary = {
            "trial": best_mlp_result["trial"],
            "decision_rule": best_mlp_result["decision_rule"],
            "calibration_metrics": best_mlp_result["calibration_metrics"],
            "test_metrics": best_mlp_result["test_metrics"],
        }
    else:
        print("Reusing existing MLP artifacts for Bayesian-only submission mode")
        mlp_summary = load_existing_mlp_summary(args.artifact_dir)

    rscript = find_rscript(args.rscript)
    runtime = bayesian_runtime(args.profile)
    bayesian_results: list[dict[str, object]] = []
    best_bayesian_result: dict[str, object] | None = None

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root_path = Path(tmp_root)
        for index, trial in enumerate(bayesian_candidates(args.profile), start=1):
            print(f"Running Bayesian tuning trial {trial['name']}")
            trial_dir = tmp_root_path / f"bayes_trial_{index:02d}"
            result = run_bayesian_trial(
                rscript=rscript,
                artifact_dir=args.artifact_dir,
                output_dir=trial_dir,
                trial=trial,
                runtime=runtime["search"],
                seed=args.seed + index,
                decision_mode=args.bayesian_decision_mode,
            )
            bayesian_results.append(
                {
                    "trial": trial,
                    "runtime": runtime["search"],
                    "metrics": result["metrics"],
                }
            )
            if best_bayesian_result is None or selection_key(result["metrics"]["calibration"]) > selection_key(
                best_bayesian_result["metrics"]["calibration"]
            ):
                best_bayesian_result = result

        assert best_bayesian_result is not None
        print(f"Selected best Bayesian trial: {best_bayesian_result['trial']['name']}")
        print("Training final Bayesian model with the selected configuration")
        final_bayesian = run_bayesian_trial(
            rscript=rscript,
            artifact_dir=args.artifact_dir,
            output_dir=args.artifact_dir,
            trial=best_bayesian_result["trial"],
            runtime=runtime["final"],
            seed=args.seed,
            decision_mode=args.bayesian_decision_mode,
        )
        selected_bayesian = select_bayesian_candidate(best_bayesian_result, final_bayesian)
        if selected_bayesian["source"] == "search":
            print("Using the Bayesian search winner as the final artifact set.")
            selected_metrics = promote_bayesian_search_artifacts(best_bayesian_result, args.artifact_dir)
        else:
            print("Using the final Bayesian retrain artifact set.")
            selected_metrics = final_bayesian["metrics"]

    save_json(args.artifact_dir / "bayesian_tuning_results.json", bayesian_results)
    ensemble_summary = maybe_build_ensemble(args.artifact_dir)

    comparison_output = run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_cold_start_artifacts.py"),
            "--artifact-dir",
            str(args.artifact_dir),
        ]
    )
    comparison_report = json.loads(comparison_output)
    save_json(args.artifact_dir / "comparison_report.json", comparison_report)
    print("Saved final comparison report")

    submission_summary = {
        "profile": args.profile,
        "artifact_dir": str(args.artifact_dir),
        "split_counts": {
            "train": int(len(train_df)),
            "calibration": int(len(calib_df)),
            "test": int(len(test_df)),
        },
        "best_mlp_trial": {
            "trial": mlp_summary["trial"],
            "decision_rule": mlp_summary.get("decision_rule"),
            "calibration_metrics": mlp_summary["calibration_metrics"],
            "test_metrics": mlp_summary["test_metrics"],
        },
        "best_bayesian_trial": {
            "trial": best_bayesian_result["trial"],
            "search_runtime": runtime["search"],
            "final_runtime": runtime["final"],
            "selected_source": selected_bayesian["source"],
            "selected_runtime": selected_bayesian["runtime"],
            "decision_mode": args.bayesian_decision_mode,
            "final_metrics": selected_metrics,
        },
        "ensemble": ensemble_summary,
        "comparison": comparison_report["winner"],
    }
    save_json(args.artifact_dir / "submission_summary.json", submission_summary)

    print(json.dumps(submission_summary, indent=2))


if __name__ == "__main__":
    main()
