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
    FEATURE_COLUMNS,
    build_cold_start_training_frame,
    encode_labels,
    fit_preprocessor,
    stratified_split_indices,
    transform_features,
)
from to_fit_or_not_to_fit.manual_mlp import ManualMLPClassifier, ManualMLPConfig  # noqa: E402
from to_fit_or_not_to_fit.metrics import (  # noqa: E402
    accuracy_score,
    classification_report_text,
    confusion_matrix,
    macro_f1_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the cold-start DemoFit Co MLP and export row-aligned split "
            "artifacts for Bayesian comparison in R."
        )
    )
    parser.add_argument("--modcloth-json", type=Path, default=ROOT / "Data" / "modcloth_final_data.json")
    parser.add_argument("--rtr-json", type=Path, default=ROOT / "Data" / "renttherunway_final_data.json")
    parser.add_argument("--brand-csv", type=Path, default=ROOT / "Data" / "demo_brand_sizing.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "cold_start")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--sample-frac-modcloth", type=float, default=1.0)
    parser.add_argument("--sample-frac-rtr", type=float, default=1.0)
    parser.add_argument("--calib-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced", "sqrt_balanced"],
        default="balanced",
        help="How to weight fit classes during MLP training.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export row-aligned split artifacts without training the MLP.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def add_probability_columns(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    for idx, class_name in enumerate(CLASS_NAMES):
        clipped = np.clip(probabilities[:, idx], 1e-8, 1.0)
        output[f"mlp_p_{class_name}"] = probabilities[:, idx]
        output[f"mlp_logit_{class_name}"] = np.log(clipped)
    return output


def export_split_frame(frame: pd.DataFrame) -> pd.DataFrame:
    split_columns = ["row_id", "fit_label", *FEATURE_COLUMNS]
    return frame.loc[:, split_columns].copy()


def build_class_weights(y_train: np.ndarray, num_classes: int, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None

    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    safe_counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * safe_counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    return (weights / np.mean(weights)).astype(np.float32)


def evaluate_split(
    name: str,
    model: ManualMLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, object]:
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)
    metrics = {
        "split": name,
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": macro_f1_score(y, predictions, labels=np.arange(len(CLASS_NAMES))),
    }
    print(f"\n{name.title()} accuracy: {metrics['accuracy']:.4f}")
    print(f"{name.title()} macro-F1: {metrics['macro_f1']:.4f}")
    print(
        classification_report_text(
            y,
            predictions,
            labels=np.arange(len(CLASS_NAMES)),
            label_names=CLASS_NAMES,
        )
    )
    cm = confusion_matrix(y, predictions, labels=np.arange(len(CLASS_NAMES)))
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_frame, size_mapping = build_cold_start_training_frame(
        modcloth_json=args.modcloth_json,
        rtr_json=args.rtr_json,
        brand_chart_path=args.brand_csv,
        sample_frac_modcloth=args.sample_frac_modcloth,
        sample_frac_rtr=args.sample_frac_rtr,
        seed=args.seed,
    )
    training_frame = training_frame.reset_index(drop=True)
    training_frame.insert(0, "row_id", np.arange(len(training_frame), dtype=np.int64))
    training_frame.to_csv(args.output_dir / "cold_start_training_table.csv", index=False)
    size_mapping.to_csv(args.output_dir / "size_label_mapping.csv", index=False)

    train_idx, calib_idx, test_idx = stratified_split_indices(
        training_frame["fit_label"],
        calib_size=args.calib_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    train_df = training_frame.iloc[train_idx].reset_index(drop=True)
    calib_df = training_frame.iloc[calib_idx].reset_index(drop=True)
    test_df = training_frame.iloc[test_idx].reset_index(drop=True)

    export_split_frame(train_df).to_csv(args.output_dir / "train_split.csv", index=False)
    export_split_frame(calib_df).to_csv(args.output_dir / "calibration_split.csv", index=False)
    export_split_frame(test_df).to_csv(args.output_dir / "test_split.csv", index=False)

    preprocessor = fit_preprocessor(train_df)
    preprocessor.to_json(args.output_dir / "preprocessor.json")

    X_train = transform_features(train_df, preprocessor)
    X_calib = transform_features(calib_df, preprocessor)
    X_test = transform_features(test_df, preprocessor)
    y_train = encode_labels(train_df["fit_label"])
    y_calib = encode_labels(calib_df["fit_label"])
    y_test = encode_labels(test_df["fit_label"])

    metadata: dict[str, object] = {
        "class_names": CLASS_NAMES,
        "feature_names": preprocessor.feature_names,
        "feature_columns": FEATURE_COLUMNS,
        "class_weighting": args.class_weighting,
        "paths": {
            "preprocessor": str(args.output_dir / "preprocessor.json"),
            "train_split": str(args.output_dir / "train_split.csv"),
            "calibration_split": str(args.output_dir / "calibration_split.csv"),
            "test_split": str(args.output_dir / "test_split.csv"),
            "size_mapping": str(args.output_dir / "size_label_mapping.csv"),
        },
    }

    if args.export_only:
        metadata["export_only"] = True
        (args.output_dir / "mlp_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print("\nExported split artifacts to:")
        print(args.output_dir)
        return

    config = ManualMLPConfig(
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        seed=args.seed,
        verbose=not args.quiet,
    )
    model = ManualMLPClassifier(
        input_dim=X_train.shape[1],
        num_classes=len(CLASS_NAMES),
        config=config,
        class_weights=build_class_weights(y_train, len(CLASS_NAMES), args.class_weighting),
        class_names=CLASS_NAMES,
    )

    print("\n" + "=" * 80)
    print("Cold-start Manual NumPy MLP")
    print("=" * 80)
    print(f"Rows: train={len(train_df):,}, calib={len(calib_df):,}, test={len(test_df):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"Feature columns: {', '.join(FEATURE_COLUMNS)}")

    history = model.fit(X_train, y_train, X_calib, y_calib)
    model.save(args.output_dir / "manual_mlp_model.npz")

    calib_metrics = evaluate_split("calibration", model, X_calib, y_calib)
    test_metrics = evaluate_split("test", model, X_test, y_test)

    calib_export = add_probability_columns(calib_df, model.predict_proba(X_calib))
    test_export = add_probability_columns(test_df, model.predict_proba(X_test))
    calib_export.to_csv(args.output_dir / "mlp_calibration_predictions.csv", index=False)
    test_export.to_csv(args.output_dir / "mlp_test_predictions.csv", index=False)

    metadata.update(
        {
            "history": history,
            "calibration_metrics": calib_metrics,
            "test_metrics": test_metrics,
        }
    )
    metadata["paths"].update(
        {
            "model": str(args.output_dir / "manual_mlp_model.npz"),
            "calibration_predictions": str(args.output_dir / "mlp_calibration_predictions.csv"),
            "test_predictions": str(args.output_dir / "mlp_test_predictions.csv"),
        }
    )
    (args.output_dir / "mlp_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved artifacts to:")
    print(args.output_dir)


if __name__ == "__main__":
    main()
