parse_named_args <- function(args, defaults) {
  parsed <- defaults
  idx <- 1L

  while (idx <= length(args)) {
    key <- args[[idx]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    }
    if (idx == length(args)) {
      stop(sprintf("Missing value for argument: %s", key), call. = FALSE)
    }

    parsed[[sub("^--", "", key)]] <- args[[idx + 1L]]
    idx <- idx + 2L
  }

  parsed
}

defaults <- list(
  input = "artifacts/cold_start/mlp_calibration_predictions.csv",
  output_model = "artifacts/cold_start/bayesian_mlp_calibrator_rstan.rds",
  output_summary = "artifacts/cold_start/bayesian_mlp_calibrator_summary.txt",
  chains = "2",
  iter = "2000",
  warmup = "1000",
  seed = "42"
)

parsed <- parse_named_args(commandArgs(trailingOnly = TRUE), defaults)

if (!requireNamespace("brms", quietly = TRUE)) {
  stop("Package 'brms' is required. Install it with install.packages('brms').", call. = FALSE)
}
if (!requireNamespace("rstan", quietly = TRUE)) {
  stop("Package 'rstan' is required. Install it with install.packages('rstan').", call. = FALSE)
}

chains <- as.integer(parsed$chains)
iter <- as.integer(parsed$iter)
warmup <- as.integer(parsed$warmup)
seed <- as.integer(parsed$seed)

options(mc.cores = max(1L, min(chains, parallel::detectCores(logical = FALSE))))
rstan::rstan_options(auto_write = TRUE)

calibration_df <- utils::read.csv(parsed$input, stringsAsFactors = FALSE)
calibration_df$fit_label <- factor(
  calibration_df$fit_label,
  levels = c("small", "fit", "large")
)
calibration_df$source <- factor(
  calibration_df$source,
  levels = c("modcloth", "renttherunway")
)
calibration_df$garment_type <- factor(
  calibration_df$garment_type,
  levels = c("tops", "bottoms", "dresses")
)
calibration_df$selected_size_label <- factor(
  calibration_df$selected_size_label,
  levels = c("XS", "S", "M", "L", "XL")
)

calibration_formula <- stats::as.formula(
  "fit_label ~ mlp_logit_small + mlp_logit_fit + mlp_logit_large + source + garment_type + selected_size_label + size_order + height_gap + bust_gap + waist_gap + hips_gap + abs_height_gap + abs_bust_gap + abs_waist_gap + abs_hips_gap + height_missing + bust_missing + waist_missing + hips_missing"
)

priors <- c(
  brms::set_prior("normal(0, 1.5)", class = "b", dpar = "mufit"),
  brms::set_prior("normal(0, 1.5)", class = "b", dpar = "mularge"),
  brms::set_prior("normal(0, 2.0)", class = "Intercept", dpar = "mufit"),
  brms::set_prior("normal(0, 2.0)", class = "Intercept", dpar = "mularge")
)

fit <- brms::brm(
  formula = calibration_formula,
  data = calibration_df,
  family = brms::categorical(),
  prior = priors,
  backend = "rstan",
  chains = chains,
  iter = iter,
  warmup = warmup,
  seed = seed,
  control = list(adapt_delta = 0.95)
)

dir.create(dirname(parsed$output_model), recursive = TRUE, showWarnings = FALSE)
saveRDS(fit, parsed$output_model)

summary_text <- capture.output({
  print(summary(fit))
  cat("\nFormula:\n")
  print(calibration_formula)
  cat("\nInput rows:", nrow(calibration_df), "\n")
})
writeLines(summary_text, parsed$output_summary)

cat("\nBayesian MLP calibrator trained with brms + rstan.\n")
cat(sprintf("Rows: %d\n", nrow(calibration_df)))
cat(sprintf("Model path: %s\n", parsed$output_model))
cat(sprintf("Summary path: %s\n", parsed$output_summary))
