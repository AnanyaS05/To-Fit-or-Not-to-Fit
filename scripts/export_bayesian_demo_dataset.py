from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from to_fit_or_not_to_fit import load_and_clean_modcloth  # noqa: E402

SIZE_LABELS = ["XS", "S", "M", "L", "XL"]
GARMENT_TYPES = ["tops", "bottoms", "dresses"]
MEASUREMENT_SPECS = {
    "height_inches": ("height_min_inches", "height_max_inches"),
    "bust": ("bust_min", "bust_max"),
    "waist": ("waist_min", "waist_max"),
    "hips": ("hips_min", "hips_max"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Bayesian-friendly training table by aligning ModCloth reviews to the "
            "demo brand size chart."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=ROOT / "Data" / "modcloth_final_data.json",
        help="Path to the ModCloth JSONL file.",
    )
    parser.add_argument(
        "--brand-csv",
        type=Path,
        default=ROOT / "Data" / "demo_brand_sizing.csv",
        help="Path to the demo brand size chart CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "Data" / "bayesian_demo_training.csv",
        help="Where to write the engineered training table.",
    )
    return parser.parse_args()


def assign_size_labels(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for garment_type, group in frame.groupby("garment_type", sort=False):
        ranked_sizes = group["size"].rank(method="first")
        labeled = pd.qcut(ranked_sizes, q=len(SIZE_LABELS), labels=SIZE_LABELS)

        group = group.copy()
        group["size_label"] = labeled.astype(str)
        group["size_ord"] = group["size_label"].map(
            {label: idx + 1 for idx, label in enumerate(SIZE_LABELS)}
        )
        pieces.append(group)

    return pd.concat(pieces).sort_index()


def build_training_table(modcloth_df: pd.DataFrame, brand_chart: pd.DataFrame) -> pd.DataFrame:
    filtered = modcloth_df[modcloth_df["category"].isin(GARMENT_TYPES)].copy()
    filtered["garment_type"] = filtered["category"]
    filtered["is_fit"] = filtered["fit"].eq("fit").astype(int)
    filtered = assign_size_labels(filtered)

    merged = filtered.merge(
        brand_chart,
        on=["garment_type", "size_label"],
        how="inner",
        validate="many_to_one",
    )

    for measure_name, (min_col, max_col) in MEASUREMENT_SPECS.items():
        center_col = f"{measure_name}_chart_center"
        missing_col = f"{measure_name}_missing"
        gap_col = f"{measure_name}_gap"
        abs_gap_col = f"abs_{measure_name}_gap"

        merged[center_col] = (merged[min_col] + merged[max_col]) / 2.0
        merged[missing_col] = merged[measure_name].isna().astype(int)
        observed = merged[measure_name].fillna(merged[center_col])
        merged[gap_col] = observed - merged[center_col]
        merged[abs_gap_col] = merged[gap_col].abs()

    selected_columns = [
        "is_fit",
        "fit",
        "garment_type",
        "size_label",
        "size_ord",
        "size",
        "brand",
        "height_inches_gap",
        "bust_gap",
        "waist_gap",
        "hips_gap",
        "abs_height_inches_gap",
        "abs_bust_gap",
        "abs_waist_gap",
        "abs_hips_gap",
        "height_inches_missing",
        "bust_missing",
        "waist_missing",
        "hips_missing",
    ]
    training_table = merged[selected_columns].rename(
        columns={
            "fit": "fit_label",
            "size": "raw_size",
            "height_inches_gap": "height_gap",
            "abs_height_inches_gap": "abs_height_gap",
            "height_inches_missing": "height_missing",
        }
    )

    return training_table


def main() -> None:
    args = parse_args()

    modcloth_df = load_and_clean_modcloth(str(args.input_json))
    brand_chart = pd.read_csv(args.brand_csv)

    training_table = build_training_table(modcloth_df, brand_chart)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    training_table.to_csv(args.output_csv, index=False)

    print(f"Saved Bayesian training table to {args.output_csv}")
    print(f"Rows: {len(training_table):,}")
    print(f"Columns: {len(training_table.columns)}")
    print("\nCounts by garment_type and size_label:")
    print(
        training_table.groupby(["garment_type", "size_label"]).size().to_string()
    )
    print("\nFit rate by garment_type:")
    print(
        training_table.groupby("garment_type")["is_fit"].mean().round(4).to_string()
    )


if __name__ == "__main__":
    main()
