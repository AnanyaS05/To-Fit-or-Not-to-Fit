# To Fit or Not to Fit

This repo explores clothing fit prediction on the ModCloth and Rent the Runway datasets.

Current project assets:
- `eda.ipynb`: exploratory data analysis and dataset profiling
- `models.ipynb`: notebook experiments with neural baselines
- `scripts/train_manual_numpy_mlp.py`: manual NumPy MLP pipeline aligned with the initial project plan
- `scripts/train_cold_start_mlp.py`: trains the cold-start MLP and exports calibration data
- `scripts/train_brms_calibrator.R`: trains the `brms`/`rstan` Bayesian calibrator
- `scripts/predict_cold_start_fit.py`: returns selected-size warnings and recommended sizes as JSON
- `Data/demo_brand_sizing.csv`: demo size chart for `XS` through `XL` across `tops`, `bottoms`, and `dresses`

Quick start:

```powershell
python scripts/train_manual_numpy_mlp.py --dataset modcloth --sample-frac-modcloth 0.25 --epochs 8
python scripts/train_manual_numpy_mlp.py --dataset renttherunway --sample-frac-rtr 0.20 --epochs 8
python scripts/train_manual_numpy_mlp.py --dataset both --task binary_misfit --epochs 10
```

Bayesian cold-start flow:

```powershell
python scripts/train_cold_start_mlp.py --epochs 25 --hidden-dims 256 128 --dropout 0.15
Rscript scripts/train_brms_calibrator.R --input artifacts/cold_start/mlp_calibration_predictions.csv --output_model artifacts/cold_start/bayesian_mlp_calibrator_rstan.rds --model-type warning_direction --formula-preset compact --weighting balanced --max-rows 10000 --chains 2 --iter 2000 --warmup 1000
python scripts/predict_cold_start_fit.py --garment-type dresses --selected-size M --height-inches 65 --bust 36 --waist 28 --hips 38
```

Faster development run:

```powershell
python scripts/train_cold_start_mlp.py --sample-frac-modcloth 0.10 --sample-frac-rtr 0.10 --epochs 20 --hidden-dims 256 128 --dropout 0.15 --output-dir artifacts/cold_start_tuned
Rscript scripts/train_brms_calibrator.R --input artifacts/cold_start_tuned/mlp_calibration_predictions.csv --output_model artifacts/cold_start_tuned/bayesian_mlp_calibrator_rstan.rds --model-type warning_direction --formula-preset compact --weighting balanced --chains 2 --iter 500 --warmup 250
python scripts/predict_cold_start_fit.py --artifact-dir artifacts/cold_start_tuned --garment-type dresses --selected-size M --height-inches 65 --bust 36 --waist 28 --hips 38
```

Use `--calibrator-model path\to\model.rds` with `predict_cold_start_fit.py` if you want to score with a calibrator file outside the artifact directory.

Validate generated artifacts:

```powershell
Rscript scripts/score_brms_calibrator.R --model artifacts/cold_start/bayesian_mlp_calibrator_rstan.rds --input artifacts/cold_start/mlp_test_predictions.csv --output artifacts/cold_start/calibrated_test_predictions.json
python scripts/validate_cold_start_artifacts.py --artifact-dir artifacts/cold_start
```

What the manual script does:
- loads and cleans the raw JSONL data
- builds train/validation/test splits
- imputes and standardizes numeric features
- one-hot encodes categorical features
- trains a manual MLP in NumPy with ReLU, dropout, Adam, class weighting, and early stopping
- reports test accuracy, macro-F1, and a confusion matrix

What the Bayesian cold-start flow does:
- uses ModCloth and Rent the Runway reviews that can be mapped into `tops`, `bottoms`, and `dresses`
- maps raw numeric source sizes into demo labels `XS`, `S`, `M`, `L`, and `XL`
- joins reviews to the demo size chart and converts measurements into gaps from the selected size
- trains a from-scratch NumPy MLP to predict `small`, `fit`, or `large`
- trains a `brms`/`rstan` Bayesian calibration model on held-out MLP predictions
- defaults to the warning-direction calibrator: one Bayesian Bernoulli model estimates `P(misfit)` and another estimates `P(large | misfit)`, then the system reconstructs `P(small)`, `P(fit)`, and `P(large)`
- supports the original categorical model with `--model-type categorical`
- scores all demo sizes and warns only when calibrated `P(small) + P(large) >= 0.70`

Notes:
- The MLP model code uses NumPy/Pandas and local metrics instead of pre-built ML models.
- R requires `brms`, `rstan`, `posterior`, `jsonlite`, `rmarkdown`, `knitr`, and a working Rtools toolchain on Windows.
- Quick R verification: `Rscript -e "library(rstan); library(brms); cat(Sys.which('make'), Sys.which('g++'), sep='\n')"`
- The final full Bayesian run should use more iterations than the fast development run. If `rstan` reports low ESS or high R-hat, increase `--iter` and `--warmup`.
