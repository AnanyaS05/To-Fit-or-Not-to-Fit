from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
TargetMode = Literal["multiclass_fit", "binary_misfit"]


@dataclass(slots=True)
class PreparedDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    num_cols: list[str]
    cat_cols: list[str]


def convert_height_modcloth(height_str: object) -> float:
    if pd.isna(height_str):
        return np.nan
    try:
        parts = str(height_str).split("ft")
        feet = int(parts[0].strip())
        inches = 0
        if len(parts) > 1 and "in" in parts[1]:
            inches = int(parts[1].replace("in", "").strip())
        value = feet * 12 + inches
        return value if 48 <= value <= 84 else np.nan
    except Exception:
        return np.nan


def convert_height_rtr(height_str: object) -> float:
    if pd.isna(height_str):
        return np.nan
    try:
        match = re.match(r"(\d+)'\s*(\d*)", str(height_str))
        if not match:
            return np.nan
        feet = int(match.group(1))
        inches = int(match.group(2)) if match.group(2) else 0
        value = feet * 12 + inches
        return value if 48 <= value <= 84 else np.nan
    except Exception:
        return np.nan


def load_and_clean_modcloth(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    df["height_inches"] = df["height"].apply(convert_height_modcloth)
    df["bust"] = pd.to_numeric(
        df["bust"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    for col in ["waist", "hips", "bra size", "shoe size", "quality"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cup size"] = df["cup size"].astype(str).str.lower().str.strip()
    df["cup size"] = df["cup size"].replace({"nan": np.nan})
    df["cup size"] = df["cup size"].str.split("/").str[0]

    keep_cols = [
        "fit",
        "size",
        "quality",
        "category",
        "length",
        "cup size",
        "height_inches",
        "waist",
        "hips",
        "bra size",
        "bust",
        "shoe size",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols].copy()

    df = df[df["fit"].isin(["small", "fit", "large"])].copy()
    df.dropna(subset=["fit", "size"], inplace=True)
    return df


def load_and_clean_rtr(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    df["height_inches"] = df["height"].apply(convert_height_rtr)
    df["weight_lbs"] = pd.to_numeric(
        df["weight"].astype(str).str.replace("lbs", "", regex=False).str.strip(),
        errors="coerce",
    )
    df["bra_size"] = pd.to_numeric(
        df["bust size"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    df["cup_size"] = df["bust size"].astype(str).str.extract(r"(\d+)([a-zA-Z+]+)$")[1]
    df["review_date"] = pd.to_datetime(
        df["review_date"],
        format="%B %d, %Y",
        errors="coerce",
    )
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 14) | (df["age"] > 90), "age"] = np.nan

    keep_cols = [
        "fit",
        "size",
        "rating",
        "category",
        "rented for",
        "body type",
        "cup_size",
        "height_inches",
        "weight_lbs",
        "age",
        "bra_size",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols].copy()

    df = df[df["fit"].isin(["small", "fit", "large"])].copy()
    df.dropna(subset=["fit", "size"], inplace=True)
    return df


def sample_dataframe(
    df: pd.DataFrame,
    frac: float | None,
    random_state: int = DEFAULT_SEED,
) -> pd.DataFrame:
    if frac is None or frac >= 1.0:
        return df.copy()
    if frac <= 0:
        raise ValueError("sample fraction must be greater than 0")
    return df.sample(frac=frac, random_state=random_state).copy()


def _build_target(df: pd.DataFrame, target_mode: TargetMode) -> pd.Series:
    if target_mode == "multiclass_fit":
        return df["fit"].copy()
    if target_mode == "binary_misfit":
        target = np.where(df["fit"].eq("fit"), "fit", "misfit")
        return pd.Series(target, index=df.index, name="misfit")
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def _ordered_class_names(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> list[str]:
    observed = set(pd.concat([y_train, y_val, y_test], ignore_index=True).astype(str).unique())

    if observed.issubset({"small", "fit", "large"}):
        return [label for label in ["small", "fit", "large"] if label in observed]
    if observed.issubset({"fit", "misfit"}):
        return [label for label in ["fit", "misfit"] if label in observed]
    return sorted(observed)


def _stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    labels = y.astype(str).to_numpy()
    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in pd.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)

        if len(indices) <= 1:
            n_test = 0
        else:
            n_test = int(round(len(indices) * test_size))
            n_test = min(max(n_test, 1), len(indices) - 1)

        test_indices.extend(indices[:n_test].tolist())
        train_indices.extend(indices[n_test:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X.iloc[train_indices].copy(),
        X.iloc[test_indices].copy(),
        y.iloc[train_indices].copy(),
        y.iloc[test_indices].copy(),
    )


def split_features_and_target(
    df: pd.DataFrame,
    target_mode: TargetMode = "multiclass_fit",
    random_state: int = DEFAULT_SEED,
    test_size: float = 0.20,
    val_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=["fit"]).copy()
    y = _build_target(df, target_mode)

    X_train_full, X_test, y_train_full, y_test = _stratified_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = _stratified_train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _prepare_numeric_matrices(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not num_cols:
        return (
            np.zeros((len(X_train), 0), dtype=np.float32),
            np.zeros((len(X_val), 0), dtype=np.float32),
            np.zeros((len(X_test), 0), dtype=np.float32),
            [],
        )

    train_num = X_train[num_cols].copy()
    val_num = X_val[num_cols].copy()
    test_num = X_test[num_cols].copy()

    medians = train_num.median()
    train_num = train_num.fillna(medians)
    val_num = val_num.fillna(medians)
    test_num = test_num.fillna(medians)

    means = train_num.mean()
    stds = train_num.std(ddof=0).replace(0, 1.0).fillna(1.0)

    train_scaled = ((train_num - means) / stds).to_numpy(dtype=np.float32)
    val_scaled = ((val_num - means) / stds).to_numpy(dtype=np.float32)
    test_scaled = ((test_num - means) / stds).to_numpy(dtype=np.float32)

    return train_scaled, val_scaled, test_scaled, num_cols.copy()


def _encode_categorical_frame(
    frame: pd.DataFrame,
    cat_cols: list[str],
    categories_by_col: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    if not cat_cols:
        return np.zeros((len(frame), 0), dtype=np.float32), []

    encoded_frames: list[pd.DataFrame] = []
    feature_names: list[str] = []

    for col in cat_cols:
        categories = categories_by_col[col]
        values = frame[col].fillna("Unknown").astype(str)
        values = values.where(values.isin(categories), "Unknown")
        categorical = pd.Categorical(values, categories=categories)
        encoded = pd.get_dummies(categorical, dtype=np.float32)
        encoded.columns = [f"{col}={category}" for category in categories]
        encoded_frames.append(encoded)
        feature_names.extend(encoded.columns.tolist())

    full_encoded = pd.concat(encoded_frames, axis=1)
    return full_encoded.to_numpy(dtype=np.float32), feature_names


def prepare_one_hot_dataset(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> PreparedDataset:
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [col for col in X_train.columns if col not in num_cols]

    X_train_num, X_val_num, X_test_num, num_feature_names = _prepare_numeric_matrices(
        X_train,
        X_val,
        X_test,
        num_cols,
    )

    categories_by_col: dict[str, list[str]] = {}
    for col in cat_cols:
        categories = X_train[col].fillna("Unknown").astype(str).unique().tolist()
        if "Unknown" not in categories:
            categories.append("Unknown")
        categories_by_col[col] = categories

    X_train_cat, cat_feature_names = _encode_categorical_frame(X_train, cat_cols, categories_by_col)
    X_val_cat, _ = _encode_categorical_frame(X_val, cat_cols, categories_by_col)
    X_test_cat, _ = _encode_categorical_frame(X_test, cat_cols, categories_by_col)

    class_names = _ordered_class_names(y_train, y_val, y_test)
    class_to_index = {label: idx for idx, label in enumerate(class_names)}

    y_train_encoded = y_train.map(class_to_index).to_numpy(dtype=np.int64)
    y_val_encoded = y_val.map(class_to_index).to_numpy(dtype=np.int64)
    y_test_encoded = y_test.map(class_to_index).to_numpy(dtype=np.int64)

    X_train_matrix = np.hstack([X_train_num, X_train_cat]).astype(np.float32, copy=False)
    X_val_matrix = np.hstack([X_val_num, X_val_cat]).astype(np.float32, copy=False)
    X_test_matrix = np.hstack([X_test_num, X_test_cat]).astype(np.float32, copy=False)

    return PreparedDataset(
        X_train=X_train_matrix,
        X_val=X_val_matrix,
        X_test=X_test_matrix,
        y_train=y_train_encoded,
        y_val=y_val_encoded,
        y_test=y_test_encoded,
        feature_names=num_feature_names + cat_feature_names,
        class_names=class_names,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )


def build_prepared_dataset(
    df: pd.DataFrame,
    target_mode: TargetMode = "multiclass_fit",
    random_state: int = DEFAULT_SEED,
    test_size: float = 0.20,
    val_size: float = 0.20,
) -> PreparedDataset:
    X_train, X_val, X_test, y_train, y_val, y_test = split_features_and_target(
        df,
        target_mode=target_mode,
        random_state=random_state,
        test_size=test_size,
        val_size=val_size,
    )
    return prepare_one_hot_dataset(X_train, X_val, X_test, y_train, y_val, y_test)
