args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- dirname(normalizePath(script_path))
source(file.path(script_dir, "bayesian_probit_utils.R"))

defaults <- list(
  input = "Data/bayesian_demo_training.csv",
  output_model = "artifacts/bayesian_probit_model.rds",
  output_summary = "artifacts/bayesian_probit_coefficients.csv",
  burn_in = "1000",
  samples = "2000",
  thin = "2",
  prior_var = "25",
  seed = "42"
)

parsed <- parse_named_args(commandArgs(trailingOnly = TRUE), defaults)

input_path <- parsed$input
output_model <- parsed$output_model
output_summary <- parsed$output_summary
burn_in <- as.integer(parsed$burn_in)
samples <- as.integer(parsed$samples)
thin <- as.integer(parsed$thin)
prior_var <- as.numeric(parsed$prior_var)
seed <- as.integer(parsed$seed)

set.seed(seed)

training_df <- utils::read.csv(input_path, stringsAsFactors = FALSE)
design <- build_design_matrix(training_df)

fit <- fit_bayesian_probit(
  X = design$X,
  y = design$y,
  burn_in = burn_in,
  samples = samples,
  thin = thin,
  prior_var = prior_var
)

posterior_mean_prob <- stats::pnorm(as.numeric(design$X %*% fit$posterior_mean))
predicted_label <- ifelse(posterior_mean_prob >= 0.5, 1L, 0L)
accuracy <- mean(predicted_label == design$y)
precision <- if (sum(predicted_label == 1L) == 0) {
  0
} else {
  mean(design$y[predicted_label == 1L] == 1L)
}
recall <- if (sum(design$y == 1L) == 0) {
  0
} else {
  mean(predicted_label[design$y == 1L] == 1L)
}

dir.create(dirname(output_model), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_summary), recursive = TRUE, showWarnings = FALSE)

coefficient_summary <- data.frame(
  feature = colnames(design$X),
  posterior_mean = fit$posterior_mean,
  posterior_sd = fit$posterior_sd,
  stringsAsFactors = FALSE
)
utils::write.csv(coefficient_summary, output_summary, row.names = FALSE)

model_payload <- list(
  beta_draws = fit$beta_draws,
  posterior_mean = fit$posterior_mean,
  feature_names = colnames(design$X),
  numeric_stats = design$numeric_stats,
  garment_levels = GARMENT_LEVELS,
  size_levels = SIZE_LEVELS,
  continuous_features = CONTINUOUS_FEATURES,
  training_metrics = list(
    accuracy = accuracy,
    precision = precision,
    recall = recall
  ),
  training_rows = nrow(training_df),
  seed = seed,
  prior_var = prior_var
)

saveRDS(model_payload, output_model)

cat("\nBayesian probit model trained.\n")
cat(sprintf("Rows: %d\n", nrow(training_df)))
cat(sprintf("Features: %d\n", ncol(design$X)))
cat(sprintf("Posterior draws saved: %d\n", nrow(fit$beta_draws)))
cat(sprintf("Training accuracy: %.4f\n", accuracy))
cat(sprintf("Training precision: %.4f\n", precision))
cat(sprintf("Training recall: %.4f\n", recall))
cat(sprintf("Model path: %s\n", output_model))
cat(sprintf("Coefficient summary path: %s\n", output_summary))
