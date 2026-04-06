from __future__ import annotations

import argparse
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
    SIZE_LABELS,
    Preprocessor,
    build_candidate_rows,
    read_brand_chart,
    transform_features,
)
from to_fit_or_not_to_fit.manual_mlp import ManualMLPClassifier  # noqa: E402

WARNING_THRESHOLD = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict cold-start fit warnings and size recommendations for DemoFit Co."
        )
    )
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "cold_start")
    parser.add_argument(
        "--calibrator-model",
        type=Path,
        default=None,
        help="Optional path to the saved brms calibrator .rds file.",
    )
    parser.add_argument("--brand-csv", type=Path, default=ROOT / "Data" / "demo_brand_sizing.csv")
    parser.add_argument("--garment-type", choices=["tops", "bottoms", "dresses"], required=True)
    parser.add_argument("--selected-size", choices=SIZE_LABELS, required=True)
    parser.add_argument("--height-inches", type=float, required=True)
    parser.add_argument("--bust", type=float, default=None)
    parser.add_argument("--waist", type=float, default=None)
    parser.add_argument("--hips", type=float, default=None)
    parser.add_argument("--rscript", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Write candidate CSV / calibrated JSON into artifact-dir instead of a temp directory.",
    )
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


def add_mlp_outputs(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    for idx, class_name in enumerate(CLASS_NAMES):
        clipped = np.clip(probabilities[:, idx], 1e-8, 1.0)
        output[f"mlp_p_{class_name}"] = probabilities[:, idx]
        output[f"mlp_logit_{class_name}"] = np.log(clipped)
    return output


def run_r_calibrator(
    rscript: str,
    model_path: Path,
    candidate_csv: Path,
    calibrated_json: Path,
) -> dict[str, object]:
    command = [
        rscript,
        str(ROOT / "scripts" / "score_brms_calibrator.R"),
        "--model",
        str(model_path),
        "--input",
        str(candidate_csv),
        "--output",
        str(calibrated_json),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "R calibration scoring failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    return json.loads(calibrated_json.read_text(encoding="utf-8"))


def aggregate_source_probabilities(payload: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in payload["rows"]:
        probabilities = row["probabilities"]
        rows.append(
            {
                "garment_type": row["garment_type"],
                "selected_size_label": row["selected_size_label"],
                "p_small": float(probabilities["small"]),
                "p_fit": float(probabilities["fit"]),
                "p_large": float(probabilities["large"]),
            }
        )

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["garment_type", "selected_size_label"], as_index=False)[
            ["p_small", "p_fit", "p_large"]
        ]
        .mean()
        .copy()
    )
    grouped["size_order"] = grouped["selected_size_label"].map(
        {label: idx for idx, label in enumerate(SIZE_LABELS)}
    )
    grouped.sort_values("size_order", inplace=True)
    return grouped


def build_response(
    garment_type: str,
    selected_size: str,
    aggregated: pd.DataFrame,
) -> dict[str, object]:
    selected = aggregated[aggregated["selected_size_label"] == selected_size]
    if selected.empty:
        raise ValueError(f"Selected size {selected_size} was not present in calibrated scores.")
    selected_row = selected.iloc[0]

    selected_probabilities = {
        "small": float(selected_row["p_small"]),
        "fit": float(selected_row["p_fit"]),
        "large": float(selected_row["p_large"]),
    }
    predicted_class = max(selected_probabilities, key=selected_probabilities.get)
    predicted_confidence = selected_probabilities[predicted_class]
    misfit_probability = selected_probabilities["small"] + selected_probabilities["large"]
    warning_direction = (
        "small"
        if selected_probabilities["small"] >= selected_probabilities["large"]
        else "large"
    )
    should_warn = misfit_probability >= WARNING_THRESHOLD

    recommendation_row = aggregated.sort_values("p_fit", ascending=False).iloc[0]
    per_size = [
        {
            "size": row["selected_size_label"],
            "probabilities": {
                "small": float(row["p_small"]),
                "fit": float(row["p_fit"]),
                "large": float(row["p_large"]),
            },
        }
        for _, row in aggregated.iterrows()
    ]

    return {
        "garment_type": garment_type,
        "selected_size": selected_size,
        "selected_size_probabilities": selected_probabilities,
        "selected_size_prediction": {
            "class": predicted_class,
            "confidence": float(predicted_confidence),
        },
        "warning": {
            "should_warn": bool(should_warn),
            "direction": warning_direction if should_warn else None,
            "confidence": float(misfit_probability),
            "threshold": WARNING_THRESHOLD,
        },
        "recommendation": {
            "size": str(recommendation_row["selected_size_label"]),
            "fit_confidence": float(recommendation_row["p_fit"]),
        },
        "per_size": per_size,
    }


def main() -> None:
    args = parse_args()

    model_path = args.artifact_dir / "manual_mlp_model.npz"
    preprocessor_path = args.artifact_dir / "preprocessor.json"
    calibrator_path = args.calibrator_model or args.artifact_dir / "bayesian_mlp_calibrator_rstan.rds"

    model = ManualMLPClassifier.load(model_path)
    preprocessor = Preprocessor.from_json(preprocessor_path)
    brand_chart = read_brand_chart(args.brand_csv)

    candidate_rows = build_candidate_rows(
        brand_chart=brand_chart,
        garment_type=args.garment_type,
        height_inches=args.height_inches,
        bust=args.bust,
        waist=args.waist,
        hips=args.hips,
        selected_sizes=SIZE_LABELS,
    )
    X_candidates = transform_features(candidate_rows, preprocessor)
    candidate_scores = add_mlp_outputs(candidate_rows, model.predict_proba(X_candidates))

    rscript = find_rscript(args.rscript)

    if args.keep_intermediate:
        work_dir = args.artifact_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        candidate_csv = work_dir / "candidate_scores_for_r.csv"
        calibrated_json = work_dir / "calibrated_candidate_scores.json"
        candidate_scores.to_csv(candidate_csv, index=False)
        payload = run_r_calibrator(rscript, calibrator_path, candidate_csv, calibrated_json)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            candidate_csv = work_dir / "candidate_scores_for_r.csv"
            calibrated_json = work_dir / "calibrated_candidate_scores.json"
            candidate_scores.to_csv(candidate_csv, index=False)
            payload = run_r_calibrator(rscript, calibrator_path, candidate_csv, calibrated_json)

    aggregated = aggregate_source_probabilities(payload)
    response = build_response(args.garment_type, args.selected_size, aggregated)
    response_text = json.dumps(response, indent=2)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(response_text, encoding="utf-8")

    print(response_text)


if __name__ == "__main__":
    main()
