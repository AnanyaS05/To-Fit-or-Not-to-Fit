from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from to_fit_or_not_to_fit import (  # noqa: E402
    DEFAULT_SEED,
    ManualMLPClassifier,
    ManualMLPConfig,
    build_prepared_dataset,
    classification_report_text,
    confusion_matrix,
    load_and_clean_modcloth,
    load_and_clean_rtr,
    sample_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the manual NumPy MLP from the initial project plan.",
    )
    parser.add_argument(
        "--dataset",
        choices=["modcloth", "renttherunway", "both"],
        default="both",
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--task",
        choices=["multiclass_fit", "binary_misfit"],
        default="multiclass_fit",
        help="Target to predict.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer sizes for the manual MLP.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sample-frac-modcloth",
        type=float,
        default=1.0,
        help="Optional sample fraction for quick ModCloth experiments.",
    )
    parser.add_argument(
        "--sample-frac-rtr",
        type=float,
        default=1.0,
        help="Optional sample fraction for quick Rent the Runway experiments.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path for saving experiment metrics as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-epoch logging.",
    )
    return parser.parse_args()


def decode_labels(encoded: np.ndarray, class_names: list[str]) -> np.ndarray:
    return np.array([class_names[index] for index in encoded], dtype=object)


def print_dataset_header(name: str, row_count: int, feature_count: int, class_names: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"Manual NumPy MLP | {name}")
    print("=" * 80)
    print(f"Rows after optional sampling: {row_count}")
    print(f"Prepared feature count: {feature_count}")
    print(f"Classes: {', '.join(class_names)}")


def run_experiment(
    dataset_name: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    prepared = build_prepared_dataset(
        df,
        target_mode=args.task,
        random_state=args.seed,
    )

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
        input_dim=prepared.X_train.shape[1],
        num_classes=len(prepared.class_names),
        config=config,
        class_names=prepared.class_names,
    )

    print_dataset_header(
        dataset_name,
        row_count=len(df),
        feature_count=prepared.X_train.shape[1],
        class_names=prepared.class_names,
    )

    history = model.fit(
        prepared.X_train,
        prepared.y_train,
        prepared.X_val,
        prepared.y_val,
    )
    metrics = model.evaluate(prepared.X_test, prepared.y_test)

    y_true_labels = decode_labels(prepared.y_test, prepared.class_names)
    y_pred_labels = decode_labels(model.predict(prepared.X_test), prepared.class_names)

    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {metrics['macro_f1']:.4f}")
    print("\nClassification report:")
    print(
        classification_report_text(
            y_true_labels,
            y_pred_labels,
            labels=np.asarray(prepared.class_names, dtype=object),
            label_names=prepared.class_names,
        )
    )

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=np.asarray(prepared.class_names, dtype=object))
    cm_df = pd.DataFrame(cm, index=prepared.class_names, columns=prepared.class_names)
    print("Confusion matrix:")
    print(cm_df.to_string())

    return {
        "dataset": dataset_name,
        "task": args.task,
        "rows": len(df),
        "feature_count": prepared.X_train.shape[1],
        "classes": prepared.class_names,
        "best_epoch": history["best_epoch"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "config": model.config_dict(),
    }


def main() -> None:
    args = parse_args()

    modcloth_df = sample_dataframe(
        load_and_clean_modcloth(str(ROOT / "Data" / "modcloth_final_data.json")),
        frac=args.sample_frac_modcloth,
        random_state=args.seed,
    )
    rtr_df = sample_dataframe(
        load_and_clean_rtr(str(ROOT / "Data" / "renttherunway_final_data.json")),
        frac=args.sample_frac_rtr,
        random_state=args.seed,
    )

    dataset_map = {
        "modcloth": ("ModCloth", modcloth_df),
        "renttherunway": ("RentTheRunway", rtr_df),
    }

    requested = ["modcloth", "renttherunway"] if args.dataset == "both" else [args.dataset]

    summaries: list[dict[str, object]] = []
    for dataset_key in requested:
        dataset_name, frame = dataset_map[dataset_key]
        summaries.append(run_experiment(dataset_name, frame, args))

    summary_df = pd.DataFrame(summaries)
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        summary_df[
            ["dataset", "task", "rows", "feature_count", "best_epoch", "accuracy", "macro_f1"]
        ].to_string(index=False)
    )

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runs": summaries}
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved metrics to {args.save_json}")


if __name__ == "__main__":
    main()
