# To Fit or Not to Fit: Final Project Report

## Overview

This project predicts whether a selected clothing size will fit as `small`, `fit`, or `large`
using customer review data from ModCloth. The final submission workflow
compares three predictors on the exact same row-aligned cold-start splits:

- a manual NumPy MLP
- a Bayesian categorical logistic regression built with `brms` and `rstan`
- a probability-blended ensemble of the two

## Data

- Raw datasets:
  - `Data/modcloth_final_data.json`
  - `Data/renttherunway_final_data.json`
- Demo size chart:
  - `Data/demo_brand_sizing.csv`
- Raw data files are intentionally excluded from git and must be placed locally under `Data/`.

The supported cold-start dataset keeps rows with missing measurements and models that
missingness directly. This is important because the Rent the Runway source would largely
disappear under complete-case filtering.

## Feature Engineering

The cold-start pipeline:

- maps source-specific size values into the demo labels `XS`, `S`, `M`, `L`, and `XL`
- aligns items to `tops`, `bottoms`, and `dresses`
- joins the selected size to the demo size chart
- converts body measurements into signed and absolute gaps from the selected size
- preserves explicit missingness indicators for height, bust, waist, and hips
- exports stable `train`, `calibration`, and `test` CSV files with `row_id`

## Models

### Manual MLP

- implemented from scratch in NumPy
- ReLU hidden layers
- dropout
- Adam optimization
- class weighting
- calibration-time bias search to improve macro-F1

### Bayesian Categorical Logistic Regression

- implemented in `brms` / `rstan`
- multiclass categorical likelihood
- compact and full formula presets available during tuning
- calibration-time bias search for macro-F1-oriented decision adjustment

### Ensemble

- postprocessing blend of saved MLP and Bayesian class probabilities
- tuned only on the calibration split
- searches a single global MLP/Bayesian blend weight
- applies the same class-bias search used by the standalone models

## Training and Evaluation Protocol

- primary metric: test macro-F1
- secondary metrics: accuracy, log-loss, and multiclass Brier score
- all three models are evaluated on the same row-aligned splits
- the ensemble uses only calibration predictions for blend and bias selection

## Final Local Submission Build

The final local artifact bundle is generated with:

```powershell
python scripts/train_submission_models.py --profile quick --artifact-dir artifacts/submission_final --sample-frac-modcloth 0.10 --sample-frac-rtr 0.10 --bayesian-decision-mode macro_f1_bias_search --rscript "C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
python scripts/validate_cold_start_artifacts.py --artifact-dir artifacts/submission_final
```

This submission package uses a sampled quick-profile build so the full pipeline remains
practical to rerun locally on the available Windows environment.

## Final Results

The current validated `artifacts/submission_final` bundle reports:

- split counts: train `14432`, calibration `4811`, test `4811`
- MLP test macro-F1: `0.37179584000988314`
- Bayesian test macro-F1: `0.3692642154892212`
- Ensemble test macro-F1: `0.371881976694675`
- winner on the primary metric: `ensemble`

The ensemble improves only slightly over the best standalone model, but it produces the
best final macro-F1 on the official local artifact bundle.

## Limitations

- the submission bundle is based on a sampled quick-profile run rather than the full raw datasets
- Bayesian final sampling is slower than the search stage, so the practical local workflow relies
  on the tuned quick profile
- minority-class performance is still much weaker than `fit`, especially for `large`

## Conclusion

The project now supports a reproducible three-model submission workflow with aligned splits,
artifact validation, and a standalone ensemble postprocessor. The ensemble is the current
best-performing submission model on the validated local final bundle.
