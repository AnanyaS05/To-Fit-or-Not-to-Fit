from __future__ import annotations

import numpy as np
import pandas as pd


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels.tolist())}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for true_label, pred_label in zip(np.asarray(y_true), np.asarray(y_pred)):
        if true_label in label_to_index and pred_label in label_to_index:
            matrix[label_to_index[true_label], label_to_index[pred_label]] += 1

    return matrix


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | np.ndarray,
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels)
    rows: list[dict[str, float | int]] = []

    for idx, label in enumerate(labels):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        support = int(cm[idx, :].sum())

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return pd.DataFrame(rows)


def macro_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | np.ndarray | None = None,
) -> float:
    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    report = precision_recall_f1(y_true, y_pred, labels)
    if report.empty:
        return 0.0
    return float(report["f1"].mean())


def classification_report_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | np.ndarray,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    report = precision_recall_f1(y_true, y_pred, labels)
    if label_names is None:
        label_names = [str(label) for label in labels]
    report = report.copy()
    report["label"] = label_names
    return report[["label", "precision", "recall", "f1", "support"]]


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | np.ndarray,
    label_names: list[str] | None = None,
    digits: int = 4,
) -> str:
    report = classification_report_dataframe(y_true, y_pred, labels, label_names)
    return report.to_string(
        index=False,
        formatters={
            "precision": lambda value: f"{value:.{digits}f}",
            "recall": lambda value: f"{value:.{digits}f}",
            "f1": lambda value: f"{value:.{digits}f}",
        },
    )
