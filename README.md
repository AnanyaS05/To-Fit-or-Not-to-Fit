# To Fit or Not to Fit

This project predicts whether a clothing item will fit as `small`, `fit`, or `large`
from structured user measurements and product metadata. It focuses on cold-start
settings where the app does not have prior purchase history for the user.

The current codebase has three main pieces:

- A Flask frontend for live probability-based fit predictions.
- A manual NumPy MLP notebook used for model development and evaluation.
- A Bayesian categorical logistic regression model in R used as an interpretable probabilistic baseline.

## Current Repository Structure

- `app.py`: Flask web app, live prediction API, frontend model loading/training, and visualization data wiring.
- `src/manual_mlp.ipynb`: manual MLP implementation from scratch in NumPy, including forward pass, backpropagation, cross-validation, and final test evaluation.
- `src/bayesian_single_model.R`: Bayesian categorical logistic regression training/evaluation with `brms` and `rstan`.
- `src/eda.ipynb`: exploratory data analysis and preprocessing work.
- `Data/train_new.csv`: shared training set used by the frontend model and Bayesian model.
- `Data/test_new.csv`: shared held-out test set used for final evaluation.
- `Data/brms_*.csv`: Bayesian CV, test summary, per-class, confusion matrix, and model metadata outputs.
- `Data/demo_brand_sizing.csv`: reference size chart used by the prediction page.
- `artifacts/frontend/frontend_manual_mlp.npz`: cached trained manual MLP used by the frontend.
- `artifacts/frontend/frontend_mlp_cv_rows.json`: cached frontend MLP-style CV metrics.
- `templates/` and `static/`: HTML, CSS, and JavaScript for the frontend.
- `Visualizations/`: exported project figures for reports, posters, and presentations.
- `Project-Documents/`: project planning and literature review documents.

## Setup

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

The app is configured for Python `3.12.10` through `runtime.txt`.

Run the frontend locally:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:10000
```

The Flask app uses Waitress and reads the `PORT` environment variable. If `PORT` is not set,
it defaults to `10000`.

## Frontend Prediction Method

The live frontend prediction route uses the trained `ManualMLPFitModel` defined in `app.py`.
At startup, the app loads the saved model from `artifacts/frontend/frontend_manual_mlp.npz`
when the cache signature matches the current data and model settings. If the cache is missing
or stale, the app trains the MLP from `Data/train_new.csv`, saves it, and then uses that saved
model for frontend predictions.

Frontend input features:

```text
size, cup size, hips, bra size, category, height
```

The prediction page lets users enter:

- garment category
- size label: `XS`, `S`, `M`, `L`, or `XL`
- hips
- height
- optional bra band size
- optional cup size letter or numeric cup code

The app maps `XS/S/M/L/XL` to a numeric size value using quantiles learned from the training set,
applies the same notebook-style MLP feature engineering, then returns probabilities for `small`,
`fit`, and `large`.

Frontend MLP feature engineering includes:

- one-hot encoded categorical columns
- numeric copies of categorical predictors
- train-set centering/scaling for numeric predictors
- selected interaction terms
- selected squared terms

Frontend model persistence:

- The live frontend uses the cached trained MLP in `artifacts/frontend/frontend_manual_mlp.npz`.
- If the training data or MLP configuration changes, the cache signature changes and the app retrains the MLP automatically.
- The Bayesian model is still used for reporting, visualization, and comparison, not live prediction.

## Model Labels And Encodings

Fit labels:

- `0`: `small`
- `1`: `fit`
- `2`: `large`

Category labels:

- `0`: `tops`
- `1`: `dresses`

Cup size letters are ordinally encoded in `app.py` using the same mapping used during preprocessing.

## Manual MLP

The manual MLP is implemented in `src/manual_mlp.ipynb`.

Key implementation locations:

- `ManualMLP` class: model initialization, forward pass, backpropagation, and parameter updates.
- `train_mlp(...)`: minibatch training loop with validation tracking and early stopping.
- `stratified_kfold_indices(...)`: stratified cross-validation split creation.
- `fit_fixed_epochs(...)`: final training loop after selecting the fixed configuration.

The notebook MLP is built from scratch using NumPy. It uses dense layers, ReLU hidden activations,
softmax output probabilities, cross-entropy loss, and Adam-style parameter updates.

Final notebook MLP test results:

- accuracy: `0.5005`
- macro-F1: `0.4555`
- small F1: `0.4213`
- fit F1: `0.6167`
- large F1: `0.3283`

## Bayesian Model

The Bayesian model is implemented in `src/bayesian_single_model.R`.

Model formula:

```r
fit ~ size + cup_size + hips + bra_size + category + height
```

Model family:

```r
categorical(link = "logit")
```

The script:

- loads `Data/train_new.csv` and `Data/test_new.csv`
- scales numeric predictors using training-set statistics
- treats `fit` as a categorical target
- runs stratified five-fold cross-validation
- trains a final Bayesian categorical logistic regression on the full training set
- predicts test labels from posterior expected class probabilities using `posterior_epred`
- writes Bayesian metrics and plots to `Data/`

Final Bayesian test results:

- accuracy: `0.4455`
- macro-F1: `0.3469`
- small F1: `0.1308`
- fit F1: `0.6037`
- large F1: `0.3061`

R dependencies are installed separately from Python:

```r
install.packages(c("brms", "rstan", "dplyr", "tibble"))
```

To rerun the Bayesian model:

```powershell
Rscript src\bayesian_single_model.R
```

Note: `src/bayesian_single_model.R` currently defines `train_path` and `test_path` near the top of
the file. If the repo is moved to a different local path, update those paths before rerunning.

## Visualizations

Saved visualization images are in `Visualizations/`.

Current static figures include:

- Bayesian confusion matrix
- Bayesian test summary metrics
- Bayesian per-class F1
- Bayesian CV metrics
- MLP confusion matrix
- Bayesian vs MLP test metric comparison
- Bayesian vs MLP CV metric comparison
- train/test class distribution

The frontend also renders interactive charts from the saved CSV/JSON artifacts.

## Data Notes

The app and final model comparison use:

- `Data/train_new.csv`
- `Data/test_new.csv`

These files contain the cleaned numeric modeling columns:

```text
size, cup size, hips, bra size, category, height, fit
```

The raw ModCloth and Rent the Runway JSON files are large and ignored by Git through:

```text
Data/*.json
```

If you need to rerun EDA from raw data, place the raw JSON files under `Data/` locally.

## Deployment Notes

- `runtime.txt` pins Python to `3.12.10`.
- `requirements.txt` contains the Python packages needed for the Flask app.
- The production entrypoint is still `python app.py`; Waitress serves the Flask app.
- On Render or similar services, make sure the app has access to `Data/train_new.csv`,
  `Data/test_new.csv`, and the required Bayesian metric CSVs if you want the visualization page
  to show the saved Bayesian comparison.

## Main Takeaways

- Structured measurements can predict clothing fit better than random guessing, even without prior user history.
- The manual MLP achieved the strongest reported notebook test performance.
- The Bayesian model is less accurate overall but gives an interpretable probabilistic baseline.
- Both model families perform best on the `fit` class and struggle more with `small` and `large`.
- The frontend turns the modeling work into a real-time fit prediction demo with class probabilities.
