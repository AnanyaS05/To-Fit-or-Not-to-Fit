from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import load_and_clean_modcloth, load_and_clean_rtr

CLASS_NAMES = ["small", "fit", "large"]
SIZE_LABELS = ["XS", "S", "M", "L", "XL"]
SOURCE_LEVELS = ["modcloth", "renttherunway"]
GARMENT_TYPES = ["tops", "bottoms", "dresses"]

MEASUREMENT_COLUMNS = ["height_inches", "bust", "waist", "hips"]
GAP_NUMERIC_COLUMNS = [
    "size_order",
    "height_gap",
    "bust_gap",
    "waist_gap",
    "hips_gap",
    "abs_height_gap",
    "abs_bust_gap",
    "abs_waist_gap",
    "abs_hips_gap",
    "height_missing",
    "bust_missing",
    "waist_missing",
    "hips_missing",
]
FEATURE_NUMERIC_COLUMNS = GAP_NUMERIC_COLUMNS
FEATURE_CATEGORICAL_COLUMNS = ["source", "garment_type", "selected_size_label"]
FEATURE_COLUMNS = FEATURE_NUMERIC_COLUMNS + FEATURE_CATEGORICAL_COLUMNS

RTR_CATEGORY_MAP = {
    "dress": "dresses",
    "gown": "dresses",
    "sheath": "dresses",
    "shift": "dresses",
    "maxi": "dresses",
    "mini": "dresses",
    "shirtdress": "dresses",
    "frock": "dresses",
    "midi": "dresses",
    "jumpsuit": "dresses",
    "romper": "dresses",
    "top": "tops",
    "blouse": "tops",
    "shirt": "tops",
    "sweater": "tops",
    "tank": "tops",
    "tunic": "tops",
    "pullover": "tops",
    "knit": "tops",
    "cardigan": "tops",
    "sweatshirt": "tops",
    "skirt": "bottoms",
    "pants": "bottoms",
    "pant": "bottoms",
    "legging": "bottoms",
    "leggings": "bottoms",
    "culotte": "bottoms",
    "culottes": "bottoms",
    "trouser": "bottoms",
}

CUP_INCHES = {
    "aa": 0.0,
    "a": 1.0,
    "b": 2.0,
    "c": 3.0,
    "d": 4.0,
    "dd": 5.0,
    "d+": 5.0,
    "ddd": 6.0,
    "e": 6.0,
    "dddd": 7.0,
    "f": 7.0,
    "g": 8.0,
    "h": 9.0,
    "i": 10.0,
    "j": 11.0,
    "k": 12.0,
}


@dataclass(slots=True)
class Preprocessor:
    numeric_cols: list[str]
    categorical_cols: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    categories_by_col: dict[str, list[str]]
    feature_names: list[str]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Preprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def normalize_cup_size(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).lower().strip().replace(" ", "")
    normalized = normalized.split("/")[0]
    normalized = normalized.replace("e", "ddd") if normalized == "e" else normalized
    return normalized or None


def estimate_bust_inches(band_size: pd.Series, cup_size: pd.Series) -> pd.Series:
    cup_offsets = cup_size.map(lambda value: CUP_INCHES.get(normalize_cup_size(value), np.nan))
    band = pd.to_numeric(band_size, errors="coerce")
    estimate = band + cup_offsets
    estimate = estimate.where(estimate.between(24, 70), np.nan)
    return estimate


def read_brand_chart(path: str | Path) -> pd.DataFrame:
    chart = pd.read_csv(path)
    chart["garment_type"] = chart["garment_type"].astype(str)
    chart["size_label"] = chart["size_label"].astype(str)
    expected = set(GARMENT_TYPES)
    observed = set(chart["garment_type"].unique())
    if not observed.issubset(expected):
        raise ValueError(f"Unexpected garment types in size chart: {sorted(observed - expected)}")
    return chart


def _sample(frame: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return frame.copy()
    if frac <= 0:
        raise ValueError("sample fractions must be greater than 0")
    return frame.sample(frac=frac, random_state=seed).copy()


def load_cold_start_sources(
    modcloth_json: str | Path,
    rtr_json: str | Path,
    sample_frac_modcloth: float = 1.0,
    sample_frac_rtr: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    modcloth = load_and_clean_modcloth(str(modcloth_json))
    modcloth = _sample(modcloth, sample_frac_modcloth, seed)
    modcloth = modcloth[modcloth["category"].isin(GARMENT_TYPES)].copy()
    modcloth["source"] = "modcloth"
    modcloth["garment_type"] = modcloth["category"]
    modcloth["fit_label"] = modcloth["fit"]
    modcloth["raw_size"] = pd.to_numeric(modcloth["size"], errors="coerce")
    modcloth["bust"] = modcloth["bust"].fillna(
        estimate_bust_inches(modcloth["bra size"], modcloth["cup size"])
    )

    rtr = load_and_clean_rtr(str(rtr_json))
    rtr = _sample(rtr, sample_frac_rtr, seed)
    rtr["source"] = "renttherunway"
    rtr["garment_type"] = rtr["category"].astype(str).str.lower().map(RTR_CATEGORY_MAP)
    rtr = rtr[rtr["garment_type"].isin(GARMENT_TYPES)].copy()
    rtr["fit_label"] = rtr["fit"]
    rtr["raw_size"] = pd.to_numeric(rtr["size"], errors="coerce")
    rtr["bust"] = estimate_bust_inches(rtr["bra_size"], rtr["cup_size"])
    rtr["waist"] = np.nan
    rtr["hips"] = np.nan

    selected_cols = [
        "source",
        "garment_type",
        "fit_label",
        "raw_size",
        "height_inches",
        "bust",
        "waist",
        "hips",
    ]
    combined = pd.concat([modcloth[selected_cols], rtr[selected_cols]], ignore_index=True)
    combined = combined[combined["fit_label"].isin(CLASS_NAMES)].copy()
    combined.dropna(subset=["raw_size"], inplace=True)
    return combined


def assign_demo_size_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    mapping_rows: list[dict[str, object]] = []
    pieces: list[pd.DataFrame] = []

    for (source, garment_type), group in frame.groupby(["source", "garment_type"], sort=False):
        if len(group) < len(SIZE_LABELS):
            continue

        ranked = group["raw_size"].rank(method="first")
        labels = pd.qcut(ranked, q=len(SIZE_LABELS), labels=SIZE_LABELS)
        labeled = group.copy()
        labeled["selected_size_label"] = labels.astype(str)
        pieces.append(labeled)

        for size_label, label_group in labeled.groupby("selected_size_label", sort=False):
            mapping_rows.append(
                {
                    "source": source,
                    "garment_type": garment_type,
                    "selected_size_label": size_label,
                    "raw_size_min": float(label_group["raw_size"].min()),
                    "raw_size_max": float(label_group["raw_size"].max()),
                    "rows": int(len(label_group)),
                }
            )

    if not pieces:
        raise ValueError("No source/garment groups had enough rows to assign demo sizes.")

    return pd.concat(pieces, ignore_index=True), pd.DataFrame(mapping_rows)


def add_size_chart_gap_features(frame: pd.DataFrame, brand_chart: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(
        brand_chart,
        left_on=["garment_type", "selected_size_label"],
        right_on=["garment_type", "size_label"],
        how="inner",
        validate="many_to_one",
    )
    merged.rename(columns={"size_order": "size_order"}, inplace=True)

    chart_bounds = {
        "height_inches": ("height_min_inches", "height_max_inches", "height"),
        "bust": ("bust_min", "bust_max", "bust"),
        "waist": ("waist_min", "waist_max", "waist"),
        "hips": ("hips_min", "hips_max", "hips"),
    }

    for measure, (min_col, max_col, output_prefix) in chart_bounds.items():
        center = (merged[min_col] + merged[max_col]) / 2.0
        missing_col = f"{output_prefix}_missing"
        gap_col = f"{output_prefix}_gap"
        abs_gap_col = f"abs_{output_prefix}_gap"

        merged[missing_col] = merged[measure].isna().astype(int)
        observed = merged[measure].fillna(center)
        merged[gap_col] = observed - center
        merged[abs_gap_col] = merged[gap_col].abs()

    return merged


def build_cold_start_training_frame(
    modcloth_json: str | Path,
    rtr_json: str | Path,
    brand_chart_path: str | Path,
    sample_frac_modcloth: float = 1.0,
    sample_frac_rtr: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    brand_chart = read_brand_chart(brand_chart_path)
    base = load_cold_start_sources(
        modcloth_json=modcloth_json,
        rtr_json=rtr_json,
        sample_frac_modcloth=sample_frac_modcloth,
        sample_frac_rtr=sample_frac_rtr,
        seed=seed,
    )
    labeled, size_mapping = assign_demo_size_labels(base)
    features = add_size_chart_gap_features(labeled, brand_chart)
    keep_cols = ["source", "garment_type", "selected_size_label", "fit_label", *FEATURE_NUMERIC_COLUMNS]
    return features[keep_cols].copy(), size_mapping


def stratified_split_indices(
    labels: Iterable[str],
    calib_size: float = 0.20,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(list(labels))
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    calib_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        n_test = int(round(len(indices) * test_size))
        n_calib = int(round(len(indices) * calib_size))
        test_indices.extend(indices[:n_test].tolist())
        calib_indices.extend(indices[n_test : n_test + n_calib].tolist())
        train_indices.extend(indices[n_test + n_calib :].tolist())

    for bucket in (train_indices, calib_indices, test_indices):
        rng.shuffle(bucket)

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(calib_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


def fit_preprocessor(frame: pd.DataFrame) -> Preprocessor:
    numeric = frame[FEATURE_NUMERIC_COLUMNS].copy()
    medians = numeric.median().fillna(0.0)
    filled = numeric.fillna(medians)
    means = filled.mean()
    stds = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)

    categories_by_col: dict[str, list[str]] = {}
    for col in FEATURE_CATEGORICAL_COLUMNS:
        categories = frame[col].fillna("Unknown").astype(str).unique().tolist()
        if col == "source":
            categories = list(dict.fromkeys([*SOURCE_LEVELS, *categories]))
        elif col == "garment_type":
            categories = list(dict.fromkeys([*GARMENT_TYPES, *categories]))
        elif col == "selected_size_label":
            categories = list(dict.fromkeys([*SIZE_LABELS, *categories]))
        if "Unknown" not in categories:
            categories.append("Unknown")
        categories_by_col[col] = categories

    feature_names = FEATURE_NUMERIC_COLUMNS.copy()
    for col in FEATURE_CATEGORICAL_COLUMNS:
        feature_names.extend([f"{col}={category}" for category in categories_by_col[col]])

    return Preprocessor(
        numeric_cols=FEATURE_NUMERIC_COLUMNS.copy(),
        categorical_cols=FEATURE_CATEGORICAL_COLUMNS.copy(),
        medians={key: float(value) for key, value in medians.items()},
        means={key: float(value) for key, value in means.items()},
        stds={key: float(value) for key, value in stds.items()},
        categories_by_col=categories_by_col,
        feature_names=feature_names,
    )


def transform_features(frame: pd.DataFrame, preprocessor: Preprocessor) -> np.ndarray:
    numeric = frame[preprocessor.numeric_cols].copy()
    for col in preprocessor.numeric_cols:
        numeric[col] = numeric[col].fillna(preprocessor.medians[col])
        numeric[col] = (numeric[col] - preprocessor.means[col]) / preprocessor.stds[col]

    matrices = [numeric.to_numpy(dtype=np.float32)]
    for col in preprocessor.categorical_cols:
        categories = preprocessor.categories_by_col[col]
        values = frame[col].fillna("Unknown").astype(str)
        values = values.where(values.isin(categories), "Unknown")
        categorical = pd.Categorical(values, categories=categories)
        matrices.append(pd.get_dummies(categorical, dtype=np.float32).to_numpy(dtype=np.float32))

    return np.hstack(matrices).astype(np.float32, copy=False)


def encode_labels(labels: pd.Series | np.ndarray) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(CLASS_NAMES)}
    return pd.Series(labels).map(mapping).to_numpy(dtype=np.int64)


def decode_labels(encoded: np.ndarray) -> np.ndarray:
    return np.asarray([CLASS_NAMES[int(index)] for index in encoded], dtype=object)


def build_candidate_rows(
    brand_chart: pd.DataFrame,
    garment_type: str,
    height_inches: float | None,
    bust: float | None,
    waist: float | None,
    hips: float | None,
    selected_sizes: list[str] | None = None,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    if garment_type not in GARMENT_TYPES:
        raise ValueError(f"garment_type must be one of {GARMENT_TYPES}")

    selected_sizes = selected_sizes or SIZE_LABELS
    sources = sources or SOURCE_LEVELS
    rows: list[dict[str, object]] = []

    for source in sources:
        for size_label in selected_sizes:
            chart_row = brand_chart[
                (brand_chart["garment_type"] == garment_type)
                & (brand_chart["size_label"] == size_label)
            ]
            if chart_row.empty:
                raise ValueError(f"No size-chart row found for {garment_type=} and {size_label=}")
            row = {
                "source": source,
                "garment_type": garment_type,
                "selected_size_label": size_label,
                "fit_label": "unknown",
                "size_order": float(chart_row.iloc[0]["size_order"]),
            }

            measurements = {
                "height": height_inches,
                "bust": bust,
                "waist": waist,
                "hips": hips,
            }
            bounds = {
                "height": ("height_min_inches", "height_max_inches"),
                "bust": ("bust_min", "bust_max"),
                "waist": ("waist_min", "waist_max"),
                "hips": ("hips_min", "hips_max"),
            }

            for name, value in measurements.items():
                min_col, max_col = bounds[name]
                center = (float(chart_row.iloc[0][min_col]) + float(chart_row.iloc[0][max_col])) / 2.0
                is_missing = value is None or pd.isna(value)
                observed = center if is_missing else float(value)
                row[f"{name}_missing"] = int(is_missing)
                row[f"{name}_gap"] = observed - center
                row[f"abs_{name}_gap"] = abs(observed - center)

            rows.append(row)

    return pd.DataFrame(rows)
