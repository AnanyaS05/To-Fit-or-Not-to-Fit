# To Fit or Not to Fit

This repo explores clothing fit prediction on the ModCloth and Rent the Runway datasets.

The supported cold-start workflow now compares two standalone multiclass models on the
same engineered data splits:

- a manual NumPy MLP
- a Bayesian categorical logistic regression built with `brms`/`rstan`

Current project assets:

- `eda.ipynb`: exploratory data analysis and dataset profiling
- `models.ipynb`: notebook experiments with neural baselines
- `scripts/train_manual_numpy_mlp.py`: manual NumPy MLP pipeline aligned with the initial project plan
- `scripts/train_cold_start_mlp.py`: trains the cold-start MLP and exports row-aligned split artifacts
- `scripts/train_submission_models.py`: tunes the submission-ready MLP/Bayesian models and auto-builds the ensemble bundle
- `scripts/train_bayesian_categorical.R`: trains the standalone Bayesian categorical classifier on the same splits
- `scripts/build_submission_ensemble.py`: rebuilds an ensemble from saved MLP/Bayesian prediction artifacts
- `scripts/validate_cold_start_artifacts.py`: compares the MLP, Bayesian model, and optional ensemble side by side
- `Data/demo_brand_sizing.csv`: demo size chart for `XS` through `XL` across `tops`, `bottoms`, and `dresses`

Quick start:

```powershell
python scripts/train_manual_numpy_mlp.py --dataset modcloth --sample-frac-modcloth 0.25 --epochs 8
python scripts/train_manual_numpy_mlp.py --dataset renttherunway --sample-frac-rtr 0.20 --epochs 8
python scripts/train_manual_numpy_mlp.py --dataset both --task binary_misfit --epochs 10
```

Environment setup:

```powershell
python -m pip install -r requirements.txt
```

Frontend app:

```powershell
python app.py
```

Then open `http://127.0.0.1:5000`.

The frontend provides:

- Landing page with project summary and model metrics
- Interactive visualization page with Bayesian CV fold metrics and class distribution
- Prediction page where users enter measurements and choose `XS/S/M/L/XL` for a `small/fit/large` prediction

R setup:

- install `brms`, `rstan`, and `jsonlite`
- see [Project-Documents/R-Environment.md](/abs/path/c:/Users/anany/Desktop/Northeastern/Year4/Spring2026/To-Fit-or-Not-to-Fit/Project-Documents/R-Environment.md)

Data placement:

- place `modcloth_final_data.json` and `renttherunway_final_data.json` under `Data/`
- these raw files are intentionally not committed; only `Data/demo_brand_sizing.csv` is tracked

Cold-start comparison flow:

```powershell
python scripts/train_cold_start_mlp.py --epochs 25 --hidden-dims 256 128 --dropout 0.15
Rscript scripts/train_bayesian_categorical.R --artifact-dir artifacts/cold_start --chains 2 --iter 2000 --warmup 1000 --weighting balanced
python scripts/validate_cold_start_artifacts.py --artifact-dir artifacts/cold_start
```

Faster development run:

```powershell
python scripts/train_cold_start_mlp.py --sample-frac-modcloth 0.10 --sample-frac-rtr 0.10 --epochs 20 --hidden-dims 256 128 --dropout 0.15 --output-dir artifacts/cold_start_tuned
Rscript scripts/train_bayesian_categorical.R --artifact-dir artifacts/cold_start_tuned --chains 2 --iter 500 --warmup 250 --weighting balanced
python scripts/validate_cold_start_artifacts.py --artifact-dir artifacts/cold_start_tuned
```

Submission orchestration flow:

```powershell
python scripts/train_submission_models.py --profile quick --artifact-dir artifacts/submission_final --sample-frac-modcloth 0.10 --sample-frac-rtr 0.10 --rscript "C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
python scripts/build_submission_ensemble.py --artifact-dir artifacts/submission_final
python scripts/validate_cold_start_artifacts.py --artifact-dir artifacts/submission_final
```

Official local submission bundle:

- artifact directory: `artifacts/submission_final`
- report summary: `artifacts/submission_final/submission_summary.json`
- comparison report: `artifacts/submission_final/comparison_report.json`
- final writeup: `Project-Documents/Final-Project-Report.md`

What the manual script does:

- loads and cleans the raw JSONL data
- builds train/validation/test splits
- imputes and standardizes numeric features
- one-hot encodes categorical features
- trains a manual MLP in NumPy with ReLU, dropout, Adam, class weighting, and early stopping
- reports test accuracy, macro-F1, and a confusion matrix

What the cold-start comparison flow does:

- uses ModCloth and Rent the Runway reviews that can be mapped into `tops`, `bottoms`, and `dresses`
- maps raw numeric source sizes into demo labels `XS`, `S`, `M`, `L`, and `XL`
- joins reviews to the demo size chart and converts measurements into gaps from the selected size
- trains a from-scratch NumPy MLP to predict `small`, `fit`, or `large`
- exports stable `train`, `calibration`, and `test` CSV splits with `row_id` so every model is evaluated on the exact same rows
- trains a standalone Bayesian categorical logistic regression on the `train` split only
- writes per-row Bayesian class probabilities for the `calibration` and `test` splits
- optionally blends saved MLP and Bayesian probabilities into an ensemble on the same row-aligned splits
- compares MLP, Bayesian, and optional ensemble performance side by side, using test macro-F1 as the primary winner metric

Notes:

- The MLP model code uses NumPy/Pandas and local metrics instead of pre-built ML models.
- R requires `brms`, `rstan`, and `jsonlite`, plus a working Rtools toolchain on Windows.
- Quick R verification: `Rscript -e "library(rstan); library(brms); cat(Sys.which('make'), Sys.which('g++'), sep='\n')"`
- The full Bayesian run should use more iterations than the fast development run. If `rstan` reports low ESS or high R-hat, increase `--iter` and `--warmup`.
