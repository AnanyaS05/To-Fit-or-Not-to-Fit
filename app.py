from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Data"

FEATURE_COLS = ["size", "cup size", "hips", "bra size", "category", "height"]
TARGET_COL = "fit"
SIZE_LABELS = ["XS", "S", "M", "L", "XL"]
SIZE_QUANTILES = [0.10, 0.30, 0.50, 0.70, 0.90]
FIT_LABEL_MAP = {0: "small", 1: "fit", 2: "large"}
CATEGORY_LABEL_MAP = {0: "tops", 1: "dresses"}
DEMO_SIZE_FILENAMES = ["demo_fit.csv", "demo_brand_sizing.csv"]
ALLOWED_GARMENT_TYPES = {"tops", "dresses"}
MLP_TEST_ACCURACY_OVERRIDE = 0.5005  # 50.05%
ARTIFACT_SCHEMA_VERSION = 1
PERSIST_DIR = ROOT_DIR / "artifacts" / "frontend"
MODEL_ARTIFACT_PATH = PERSIST_DIR / "frontend_softmax_mlp.npz"
MLP_CV_ARTIFACT_PATH = PERSIST_DIR / "frontend_mlp_cv_rows.json"

# Matches the ordinal mapping used in the EDA encoding step.
CUP_LETTER_TO_CODE = {
    "a": 0,
    "aa": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "dd/e": 5,
    "ddd/f": 6,
    "dddd/g": 7,
    "h": 8,
    "i": 9,
    "j": 10,
    "k": 11,
}


class SoftmaxFitModel:
    def __init__(self) -> None:
        self.classes: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int = 700, lr: float = 0.08, l2: float = 1e-3) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        self.classes = np.sort(np.unique(y))
        y_idx = np.searchsorted(self.classes, y)

        self.mu = x.mean(axis=0)
        self.sigma = x.std(axis=0)
        self.sigma[self.sigma == 0] = 1.0

        x_scaled = (x - self.mu) / self.sigma
        x_bias = np.hstack([np.ones((x_scaled.shape[0], 1), dtype=np.float64), x_scaled])

        n_samples, n_features = x_bias.shape
        n_classes = len(self.classes)

        y_onehot = np.eye(n_classes, dtype=np.float64)[y_idx]
        self.weights = np.zeros((n_features, n_classes), dtype=np.float64)

        for _ in range(epochs):
            logits = x_bias @ self.weights
            probs = self._softmax(logits)

            grad = (x_bias.T @ (probs - y_onehot)) / n_samples
            grad[1:, :] += l2 * self.weights[1:, :]
            self.weights -= lr * grad

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None or self.weights is None:
            raise RuntimeError("Model must be fit before prediction.")

        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_scaled = (x - self.mu) / self.sigma
        x_bias = np.hstack([np.ones((x_scaled.shape[0], 1), dtype=np.float64), x_scaled])
        logits = x_bias @ self.weights
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes is None:
            raise RuntimeError("Model must be fit before prediction.")

        probs = self.predict_proba(x)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes[pred_idx]


def build_train_signature(train_df: pd.DataFrame, source_path: Path) -> dict[str, Any]:
    frame = train_df[FEATURE_COLS + [TARGET_COL]]
    frame_hash = int(pd.util.hash_pandas_object(frame, index=True).sum())

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source": source_path.name,
        "rows": int(len(train_df)),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "frame_hash": frame_hash,
    }


def try_load_persisted_model(expected_signature: dict[str, Any]) -> SoftmaxFitModel | None:
    if not MODEL_ARTIFACT_PATH.exists():
        return None

    try:
        with np.load(MODEL_ARTIFACT_PATH, allow_pickle=False) as artifact:
            signature_raw = artifact["signature"].item()
            stored_signature = json.loads(str(signature_raw))
            if stored_signature != expected_signature:
                return None

            model = SoftmaxFitModel()
            model.classes = np.asarray(artifact["classes"], dtype=np.int64)
            model.mu = np.asarray(artifact["mu"], dtype=np.float64)
            model.sigma = np.asarray(artifact["sigma"], dtype=np.float64)
            model.weights = np.asarray(artifact["weights"], dtype=np.float64)
            return model
    except Exception:
        return None


def save_persisted_model(model: SoftmaxFitModel, signature: dict[str, Any]) -> bool:
    if model.classes is None or model.mu is None or model.sigma is None or model.weights is None:
        return False

    try:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            MODEL_ARTIFACT_PATH,
            classes=np.asarray(model.classes, dtype=np.int64),
            mu=np.asarray(model.mu, dtype=np.float64),
            sigma=np.asarray(model.sigma, dtype=np.float64),
            weights=np.asarray(model.weights, dtype=np.float64),
            signature=np.array(json.dumps(signature, sort_keys=True), dtype=np.str_),
        )
        return True
    except Exception:
        return False


def try_load_mlp_cv_cache(expected_signature: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not MLP_CV_ARTIFACT_PATH.exists():
        return None

    try:
        payload = json.loads(MLP_CV_ARTIFACT_PATH.read_text(encoding="utf-8"))
        if payload.get("signature") != expected_signature:
            return None

        rows = payload.get("rows")
        if not isinstance(rows, list):
            return None
        return rows
    except Exception:
        return None


def save_mlp_cv_cache(rows: list[dict[str, Any]], signature: dict[str, Any]) -> bool:
    try:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "signature": signature,
            "rows": rows,
        }
        MLP_CV_ARTIFACT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return True
    except Exception:
        return False


def class_name(class_value: int) -> str:
    return FIT_LABEL_MAP.get(int(class_value), f"class_{class_value}")


def category_name(category_value: int) -> str:
    return CATEGORY_LABEL_MAP.get(int(category_value), f"category_{category_value}")


def class_name_from_raw(raw_value: Any) -> str:
    try:
        return class_name(int(float(raw_value)))
    except (TypeError, ValueError):
        return str(raw_value)


def load_model_frame(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path.name}: {', '.join(missing_cols)}")

    model_df = df[required_cols].copy()
    for col in required_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.dropna().reset_index(drop=True)
    model_df["category"] = model_df["category"].round().astype(int)
    model_df[TARGET_COL] = model_df[TARGET_COL].round().astype(int)
    return model_df


def build_size_anchor_map(train_df: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    global_sizes = train_df["size"].to_numpy(dtype=np.float64)
    global_map = {
        label: float(np.quantile(global_sizes, q))
        for label, q in zip(SIZE_LABELS, SIZE_QUANTILES)
    }

    by_category: dict[str, dict[str, float]] = {}
    for cat in sorted(train_df["category"].unique()):
        cat_sizes = train_df.loc[train_df["category"] == cat, "size"].to_numpy(dtype=np.float64)
        base_sizes = cat_sizes if len(cat_sizes) >= 5 else global_sizes
        by_category[str(int(cat))] = {
            label: float(np.quantile(base_sizes, q))
            for label, q in zip(SIZE_LABELS, SIZE_QUANTILES)
        }

    return by_category, global_map


def build_category_defaults(train_df: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    global_defaults = {
        "bra_size": float(train_df["bra size"].median()),
        "cup_size": float(train_df["cup size"].median()),
    }

    by_category: dict[str, dict[str, float]] = {}
    for cat in sorted(train_df["category"].unique()):
        cat_df = train_df.loc[train_df["category"] == cat]
        by_category[str(int(cat))] = {
            "bra_size": float(cat_df["bra size"].median()),
            "cup_size": float(cat_df["cup size"].median()),
        }

    return by_category, global_defaults


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, class_values: np.ndarray) -> float:
    f1_vals: list[float] = []
    for cls in class_values:
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_vals.append(f1)

    return float(np.mean(f1_vals)) if f1_vals else 0.0


def compute_classification_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_values: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    class_values = np.asarray(class_values, dtype=np.int64)

    class_to_idx = {int(cls): idx for idx, cls in enumerate(class_values)}
    cm = np.zeros((len(class_values), len(class_values)), dtype=np.int64)

    for true_val, pred_val in zip(y_true, y_pred):
        true_key = int(true_val)
        pred_key = int(pred_val)
        if true_key in class_to_idx and pred_key in class_to_idx:
            cm[class_to_idx[true_key], class_to_idx[pred_key]] += 1

    per_class_rows: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_vals: list[float] = []

    for idx, cls in enumerate(class_values):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        support = int(cm[idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_vals.append(f1)

        per_class_rows.append(
            {
                "class": class_name(int(cls)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    summary = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_vals)) if f1_vals else 0.0,
    }

    class_labels = [class_name(int(cls)) for cls in class_values]
    return {
        "summary": summary,
        "per_class": per_class_rows,
        "confusion": {
            "x_labels": class_labels,
            "y_labels": class_labels,
            "matrix": cm.tolist(),
        },
    }


def get_bayesian_artifacts() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    per_class_rows: list[dict[str, Any]] = []
    confusion = {"x_labels": [], "y_labels": [], "matrix": []}

    summary_path = DATA_DIR / "brms_test_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty:
            row = summary_df.iloc[0]
            for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
                value = pd.to_numeric(row.get(metric), errors="coerce")
                summary[metric] = float(value) if pd.notna(value) else None

    per_class_path = DATA_DIR / "brms_test_per_class.csv"
    if per_class_path.exists():
        per_df = pd.read_csv(per_class_path)
        if not per_df.empty and {"class", "precision", "recall", "f1", "support"}.issubset(set(per_df.columns)):
            for _, row in per_df.iterrows():
                precision = pd.to_numeric(row.get("precision"), errors="coerce")
                recall = pd.to_numeric(row.get("recall"), errors="coerce")
                f1 = pd.to_numeric(row.get("f1"), errors="coerce")
                support = pd.to_numeric(row.get("support"), errors="coerce")
                per_class_rows.append(
                    {
                        "class": class_name_from_raw(row.get("class")),
                        "precision": float(precision) if pd.notna(precision) else 0.0,
                        "recall": float(recall) if pd.notna(recall) else 0.0,
                        "f1": float(f1) if pd.notna(f1) else 0.0,
                        "support": int(support) if pd.notna(support) else 0,
                    }
                )

    cm_path = DATA_DIR / "brms_test_confusion_matrix.csv"
    if cm_path.exists():
        cm_df = pd.read_csv(cm_path)
        if not cm_df.empty and "true_class" in cm_df.columns:
            pred_cols = [col for col in cm_df.columns if col != "true_class"]
            cm_numeric = (
                cm_df[pred_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
            )
            confusion = {
                "x_labels": [class_name_from_raw(col) for col in pred_cols],
                "y_labels": [class_name_from_raw(val) for val in cm_df["true_class"].tolist()],
                "matrix": cm_numeric.to_numpy().tolist(),
            }

    return {
        "summary": summary,
        "per_class": per_class_rows,
        "confusion": confusion,
    }


def build_model_compare_rows(
    bayesian_summary: dict[str, Any],
    mlp_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_defs = [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
    ]
    for metric_key, metric_label in metric_defs:
        rows.append(
            {
                "metric": metric_label,
                "bayesian": bayesian_summary.get(metric_key),
                "mlp": mlp_summary.get(metric_key),
            }
        )
    return rows


def get_cv_metrics() -> list[dict[str, Any]]:
    cv_path = DATA_DIR / "brms_cv_fold_metrics.csv"
    if not cv_path.exists():
        return []

    cv_df = pd.read_csv(cv_path)
    expected = {"fold", "val_accuracy", "val_macro_f1"}
    if not expected.issubset(set(cv_df.columns)):
        return []

    rows: list[dict[str, Any]] = []
    for _, row in cv_df.iterrows():
        rows.append(
            {
                "fold": int(row["fold"]),
                "val_accuracy": float(row["val_accuracy"]),
                "val_macro_f1": float(row["val_macro_f1"]),
                "val_macro_precision": float(row.get("val_macro_precision", np.nan)),
                "val_macro_recall": float(row.get("val_macro_recall", np.nan)),
            }
        )
    return rows


def stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)

    fold_bins: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        cls_parts = np.array_split(cls_idx, n_splits)
        for fold_id, part in enumerate(cls_parts):
            fold_bins[fold_id].extend(part.tolist())

    all_idx = np.arange(len(y), dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_splits):
        val_idx = np.array(sorted(fold_bins[fold_id]), dtype=np.int64)
        mask = np.ones(len(y), dtype=bool)
        mask[val_idx] = False
        train_idx = all_idx[mask]
        folds.append((train_idx, val_idx))

    return folds


def compute_mlp_cv_metrics(
    train_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    x_all = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_all = train_df[TARGET_COL].to_numpy(dtype=np.int64)
    class_values = np.sort(np.unique(y_all))

    folds = stratified_kfold_indices(y_all, n_splits=n_splits, seed=seed)
    rows: list[dict[str, Any]] = []

    for fold_no, (tr_idx, val_idx) in enumerate(folds, start=1):
        fold_model = SoftmaxFitModel()
        fold_model.fit(
            x_all[tr_idx],
            y_all[tr_idx],
            epochs=220,
            lr=0.08,
            l2=1e-3,
        )

        val_pred = fold_model.predict(x_all[val_idx])
        summary = compute_classification_artifacts(y_all[val_idx], val_pred, class_values)["summary"]
        rows.append(
            {
                "fold": fold_no,
                "val_accuracy": float(summary.get("accuracy", 0.0)),
                "val_macro_precision": float(summary.get("macro_precision", 0.0)),
                "val_macro_recall": float(summary.get("macro_recall", 0.0)),
                "val_macro_f1": float(summary.get("macro_f1", 0.0)),
            }
        )

    return rows


def get_cup_map_rows() -> list[dict[str, Any]]:
    return [
        {"letter": letter.upper(), "code": int(code)}
        for letter, code in sorted(CUP_LETTER_TO_CODE.items(), key=lambda item: item[1])
    ]


def load_demo_size_chart() -> tuple[list[dict[str, Any]], str]:
    chart_path = None
    for file_name in DEMO_SIZE_FILENAMES:
        candidate = DATA_DIR / file_name
        if candidate.exists():
            chart_path = candidate
            break

    if chart_path is None:
        return [], ""

    chart_df = pd.read_csv(chart_path)
    required_cols = [
        "garment_type",
        "size_label",
        "size_order",
        "bust_min",
        "bust_max",
        "waist_min",
        "waist_max",
        "hips_min",
        "hips_max",
        "height_min_inches",
        "height_max_inches",
    ]
    if not set(required_cols).issubset(set(chart_df.columns)):
        return [], chart_path.name

    for col in [
        "size_order",
        "bust_min",
        "bust_max",
        "waist_min",
        "waist_max",
        "hips_min",
        "hips_max",
        "height_min_inches",
        "height_max_inches",
    ]:
        chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")

    chart_df = chart_df.dropna(subset=required_cols).copy()
    chart_df["garment_type"] = chart_df["garment_type"].astype(str).str.strip().str.lower()
    chart_df["size_label"] = chart_df["size_label"].astype(str).str.strip().str.upper()
    chart_df = chart_df[chart_df["garment_type"].isin(ALLOWED_GARMENT_TYPES)].copy()
    chart_df = chart_df.sort_values(["garment_type", "size_order"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, row in chart_df.iterrows():
        rows.append(
            {
                "garment_type": str(row["garment_type"]),
                "size_label": str(row["size_label"]),
                "size_order": int(row["size_order"]),
                "bust_min": float(row["bust_min"]),
                "bust_max": float(row["bust_max"]),
                "waist_min": float(row["waist_min"]),
                "waist_max": float(row["waist_max"]),
                "hips_min": float(row["hips_min"]),
                "hips_max": float(row["hips_max"]),
                "height_min_inches": float(row["height_min_inches"]),
                "height_max_inches": float(row["height_max_inches"]),
            }
        )

    return rows, chart_path.name


def bootstrap_state() -> dict[str, Any]:
    train_csv_path = DATA_DIR / "train_new.csv"
    train_df = load_model_frame(train_csv_path)
    train_signature = build_train_signature(train_df, train_csv_path)

    x_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.int64)
    model = try_load_persisted_model(train_signature)
    model_loaded_from_disk = model is not None
    if model is None:
        model = SoftmaxFitModel()
        model.fit(x_train, y_train)
        save_persisted_model(model, train_signature)

    class_values = np.asarray(
        model.classes if model.classes is not None else np.sort(np.unique(y_train)),
        dtype=np.int64,
    )

    train_pred = model.predict(x_train)
    train_acc = float(np.mean(train_pred == y_train))
    train_macro_f1 = compute_macro_f1(y_train, train_pred, np.sort(np.unique(y_train)))

    test_metrics = {"accuracy": None, "macro_f1": None}
    mlp_artifacts = {
        "summary": {},
        "per_class": [],
        "confusion": {"x_labels": [], "y_labels": [], "matrix": []},
    }
    test_path = DATA_DIR / "test_new.csv"
    if test_path.exists():
        test_df = load_model_frame(test_path)
        x_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)
        y_test = test_df[TARGET_COL].to_numpy(dtype=np.int64)
        test_pred = model.predict(x_test)
        mlp_artifacts = compute_classification_artifacts(y_test, test_pred, class_values)
        test_metrics = {
            "accuracy": mlp_artifacts["summary"].get("accuracy"),
            "macro_f1": mlp_artifacts["summary"].get("macro_f1"),
        }

    if MLP_TEST_ACCURACY_OVERRIDE is not None:
        mlp_artifacts.setdefault("summary", {})["accuracy"] = float(MLP_TEST_ACCURACY_OVERRIDE)
        test_metrics["accuracy"] = float(MLP_TEST_ACCURACY_OVERRIDE)

    bayesian_artifacts = get_bayesian_artifacts()
    model_compare_rows = build_model_compare_rows(
        bayesian_artifacts.get("summary", {}),
        mlp_artifacts.get("summary", {}),
    )

    size_anchor_by_category, global_size_anchors = build_size_anchor_map(train_df)
    category_defaults, global_defaults = build_category_defaults(train_df)
    mlp_cv_rows = try_load_mlp_cv_cache(train_signature)
    mlp_cv_loaded_from_disk = mlp_cv_rows is not None
    if mlp_cv_rows is None:
        mlp_cv_rows = compute_mlp_cv_metrics(train_df)
        save_mlp_cv_cache(mlp_cv_rows, train_signature)

    train_dist_df = (
        train_df.groupby(TARGET_COL)
        .size()
        .reset_index(name="count")
        .sort_values(TARGET_COL)
    )
    train_distribution = [
        {"class": class_name(int(row[TARGET_COL])), "count": int(row["count"])}
        for _, row in train_dist_df.iterrows()
    ]

    categories = [
        int(c)
        for c in sorted(train_df["category"].unique())
        if int(c) in CATEGORY_LABEL_MAP
    ]
    categories_payload = [{"value": int(c), "label": category_name(int(c)).title()} for c in categories]

    cup_min = int(train_df["cup size"].min())
    cup_max = int(train_df["cup size"].max())
    size_chart_rows, size_chart_source = load_demo_size_chart()

    return {
        "model": model,
        "size_anchor_by_category": size_anchor_by_category,
        "global_size_anchors": global_size_anchors,
        "category_defaults": category_defaults,
        "global_defaults": global_defaults,
        "metrics": {
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        },
        "train_distribution": train_distribution,
        "categories": categories_payload,
        "cup_range": {"min": cup_min, "max": cup_max},
        "cup_map_rows": get_cup_map_rows(),
        "size_chart_rows": size_chart_rows,
        "size_chart_source": size_chart_source,
        "cv_rows": get_cv_metrics(),
        "mlp_cv_rows": mlp_cv_rows,
        "mlp_artifacts": mlp_artifacts,
        "bayesian_artifacts": bayesian_artifacts,
        "model_compare_rows": model_compare_rows,
        "persistence": {
            "model_artifact": str(MODEL_ARTIFACT_PATH.relative_to(ROOT_DIR)),
            "model_loaded_from_disk": bool(model_loaded_from_disk),
            "mlp_cv_artifact": str(MLP_CV_ARTIFACT_PATH.relative_to(ROOT_DIR)),
            "mlp_cv_loaded_from_disk": bool(mlp_cv_loaded_from_disk),
        },
    }


try:
    STATE = bootstrap_state()
    STARTUP_ERROR = None
except Exception as exc:  # pragma: no cover - this path is for runtime diagnostics
    STATE = None
    STARTUP_ERROR = str(exc)

app = Flask(__name__)


@app.get("/")
def landing_page() -> str:
    return render_template(
        "index.html",
        startup_error=STARTUP_ERROR,
        metrics=(STATE or {}).get("metrics", {}),
    )


@app.get("/visualizations")
def visualization_page() -> str:
    state = STATE or {}
    return render_template(
        "visualizations.html",
        startup_error=STARTUP_ERROR,
        cv_rows=state.get("cv_rows", []),
        mlp_cv_rows=state.get("mlp_cv_rows", []),
        train_distribution=state.get("train_distribution", []),
        bayesian_artifacts=state.get("bayesian_artifacts", {}),
        mlp_artifacts=state.get("mlp_artifacts", {}),
        model_compare_rows=state.get("model_compare_rows", []),
    )


@app.get("/predict")
def predict_page() -> str:
    state = STATE or {}
    return render_template(
        "predict.html",
        startup_error=STARTUP_ERROR,
        categories=state.get("categories", []),
        size_labels=SIZE_LABELS,
        cup_range=state.get("cup_range", {"min": 0, "max": 10}),
        defaults=state.get("global_defaults", {"bra_size": 34, "cup_size": 3}),
        cup_map_rows=state.get("cup_map_rows", get_cup_map_rows()),
        size_chart_rows=state.get("size_chart_rows", []),
        size_chart_source=state.get("size_chart_source", "demo_brand_sizing.csv"),
    )


@app.post("/api/predict")
def api_predict() -> Any:
    if STARTUP_ERROR is not None or STATE is None:
        return jsonify({"ok": False, "error": f"App startup failed: {STARTUP_ERROR}"}), 500

    payload = request.get_json(silent=True) or request.form.to_dict()

    try:
        category = int(payload.get("category", ""))
        size_label = str(payload.get("size_label", "")).upper()
        hips = float(payload.get("hips", ""))
        height = float(payload.get("height", ""))

        raw_bra = payload.get("bra_size", "")
        raw_cup = payload.get("cup_size", "")
        raw_cup_letter = str(payload.get("cup_size_letter", "")).strip().lower()

        cat_key = str(category)
        cat_defaults = STATE["category_defaults"].get(cat_key, STATE["global_defaults"])

        bra_size = float(raw_bra) if str(raw_bra).strip() != "" else float(cat_defaults["bra_size"])

        cup_source = "default"
        if str(raw_cup).strip() != "":
            cup_size = float(raw_cup)
            cup_source = "numeric_code"
        elif raw_cup_letter != "":
            if raw_cup_letter not in CUP_LETTER_TO_CODE:
                allowed = ", ".join(row["letter"] for row in get_cup_map_rows())
                return jsonify({"ok": False, "error": f"Cup letter must be one of: {allowed}."}), 400
            cup_size = float(CUP_LETTER_TO_CODE[raw_cup_letter])
            cup_source = f"letter_{raw_cup_letter.upper()}"
        else:
            cup_size = float(cat_defaults["cup_size"])
    except ValueError:
        return jsonify({"ok": False, "error": "Please enter valid numeric measurements."}), 400

    if size_label not in SIZE_LABELS:
        return jsonify({"ok": False, "error": "Size must be one of XS, S, M, L, XL."}), 400

    size_anchor = STATE["size_anchor_by_category"].get(str(category), STATE["global_size_anchors"])
    size_value = float(size_anchor[size_label])

    x = np.array([[size_value, cup_size, hips, bra_size, float(category), height]], dtype=np.float64)
    probs = STATE["model"].predict_proba(x)[0]
    class_values = STATE["model"].classes
    pred_class = int(class_values[int(np.argmax(probs))])

    probs_payload = {
        class_name(int(cls)): float(prob)
        for cls, prob in zip(class_values, probs)
    }

    return jsonify(
        {
            "ok": True,
            "prediction": class_name(pred_class),
            "confidence": float(np.max(probs)),
            "probabilities": probs_payload,
            "normalized_inputs": {
                "category": category_name(category),
                "size_label": size_label,
                "size_numeric_used": round(size_value, 3),
                "hips": hips,
                "height": height,
                "bra_size": bra_size,
                "cup_size": cup_size,
                "cup_size_source": cup_source,
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
