 # Train the Bayesian calibration layer for the cold-start fit predictor.
 #
 # The default "warning_direction" model matches the product flow:
 # 1. A Bernoulli model estimates P(selected size is a misfit).
 # 2. A second Bernoulli model estimates P(large | selected size is a misfit).
 # 3. The scorer reconstructs P(small), P(fit), and P(large).
 #
 # The original categorical calibrator is still available with:
 # --model-type categorical

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

    parsed[[gsub("-", "_", sub("^--", "", key))]] <- args[[idx + 1L]]
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
  seed = "42",
  max_rows = "0",
  model_type = "warning_direction",
  formula_preset = "compact",
  weighting = "sqrt_balanced",
  max_weight = "3",
  prior_scale = "1.0",
  intercept_prior_scale = "1.5",
  adapt_delta = "0.97",
  max_treedepth = "12"
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
max_rows <- as.integer(parsed$max_rows)
max_weight <- as.numeric(parsed$max_weight)
prior_scale <- as.numeric(parsed$prior_scale)
intercept_prior_scale <- as.numeric(parsed$intercept_prior_scale)
adapt_delta <- as.numeric(parsed$adapt_delta)
max_treedepth <- as.integer(parsed$max_treedepth)

valid_formula_presets <- c("logits_only", "compact", "gaps", "full")
if (!parsed$formula_preset %in% valid_formula_presets) {
  stop(
    sprintf(
      "--formula_preset must be one of: %s",
      paste(valid_formula_presets, collapse = ", ")
    ),
    call. = FALSE
  )
}

valid_weighting <- c("none", "balanced", "sqrt_balanced")
if (!parsed$weighting %in% valid_weighting) {
  stop(
    sprintf("--weighting must be one of: %s", paste(valid_weighting, collapse = ", ")),
    call. = FALSE
  )
}

valid_model_types <- c("categorical", "warning_direction")
if (!parsed$model_type %in% valid_model_types) {
  stop(
    sprintf("--model_type must be one of: %s", paste(valid_model_types, collapse = ", ")),
    call. = FALSE
  )
}

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

sample_stratified <- function(frame, label_col, max_rows, seed) {
  if (max_rows <= 0 || nrow(frame) <= max_rows) {
    return(frame)
  }

  set.seed(seed)
  labels <- frame[[label_col]]
  count_frame <- as.data.frame(table(labels), stringsAsFactors = FALSE)
  names(count_frame) <- c("label", "count")
  count_frame <- count_frame[count_frame$count > 0, , drop = FALSE]
  count_frame$target <- pmax(1L, floor(max_rows * count_frame$count / sum(count_frame$count)))

  # Add any rounding leftovers to the largest classes so the total reaches max_rows.
  leftover <- max_rows - sum(count_frame$target)
  if (leftover > 0) {
    order_indices <- order(count_frame$count, decreasing = TRUE)
    for (idx in order_indices[seq_len(leftover)]) {
      count_frame$target[[idx]] <- count_frame$target[[idx]] + 1L
    }
  }

  selected <- unlist(
    lapply(seq_len(nrow(count_frame)), function(row_idx) {
      label <- count_frame$label[[row_idx]]
      indices <- which(labels == label)
      sample(indices, size = min(length(indices), count_frame$target[[row_idx]]), replace = FALSE)
    }),
    use.names = FALSE
  )
  frame[sample(selected), , drop = FALSE]
}

calibration_df <- sample_stratified(calibration_df, "fit_label", max_rows, seed)

build_class_weights <- function(labels, method, max_weight) {
  if (method == "none") {
    return(rep(1.0, length(labels)))
  }

  counts <- table(labels)
  balanced <- length(labels) / (length(counts) * counts)
  if (method == "sqrt_balanced") {
    balanced <- sqrt(balanced)
  }

  weights <- as.numeric(balanced[as.character(labels)])
  weights <- weights / mean(weights)
  pmin(weights, max_weight)
}

calibration_df$calibration_weight <- build_class_weights(
  calibration_df$fit_label,
  parsed$weighting,
  max_weight
)
calibration_df$misfit_label <- as.integer(calibration_df$fit_label != "fit")
calibration_df$misfit_weight <- build_class_weights(
  factor(ifelse(calibration_df$misfit_label == 1L, "misfit", "fit"), levels = c("fit", "misfit")),
  parsed$weighting,
  max_weight
)

terms_by_preset <- list(
  logits_only = c(
    "mlp_logit_small",
    "mlp_logit_fit",
    "mlp_logit_large"
  ),
  compact = c(
    "mlp_logit_small",
    "mlp_logit_fit",
    "mlp_logit_large",
    "source",
    "garment_type",
    "selected_size_label"
  ),
  gaps = c(
    "mlp_logit_small",
    "mlp_logit_fit",
    "mlp_logit_large",
    "source",
    "garment_type",
    "selected_size_label",
    "height_gap",
    "bust_gap",
    "waist_gap",
    "hips_gap",
    "height_missing",
    "bust_missing",
    "waist_missing",
    "hips_missing"
  ),
  full = c(
    "mlp_logit_small",
    "mlp_logit_fit",
    "mlp_logit_large",
    "source",
    "garment_type",
    "selected_size_label",
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
    "hips_missing"
  )
)

base_terms <- paste(terms_by_preset[[parsed$formula_preset]], collapse = " + ")
calibration_formula <- stats::as.formula(
  sprintf(
    "fit_label | weights(calibration_weight) ~ %s",
    base_terms
  )
)
misfit_formula <- stats::as.formula(
  sprintf("misfit_label | weights(misfit_weight) ~ %s", base_terms)
)
direction_df <- calibration_df[calibration_df$fit_label != "fit", , drop = FALSE]
direction_df$large_label <- as.integer(direction_df$fit_label == "large")
direction_df$direction_weight <- build_class_weights(
  factor(ifelse(direction_df$large_label == 1L, "large", "small"), levels = c("small", "large")),
  "sqrt_balanced",
  max_weight
)
direction_formula <- stats::as.formula(
  sprintf("large_label | weights(direction_weight) ~ %s", base_terms)
)

categorical_priors <- c(
  brms::set_prior(sprintf("normal(0, %.4f)", prior_scale), class = "b", dpar = "mufit"),
  brms::set_prior(sprintf("normal(0, %.4f)", prior_scale), class = "b", dpar = "mularge"),
  brms::set_prior(
    sprintf("normal(0, %.4f)", intercept_prior_scale),
    class = "Intercept",
    dpar = "mufit"
  ),
  brms::set_prior(
    sprintf("normal(0, %.4f)", intercept_prior_scale),
    class = "Intercept",
    dpar = "mularge"
  )
)

bernoulli_priors <- c(
  brms::set_prior(sprintf("normal(0, %.4f)", prior_scale), class = "b"),
  brms::set_prior(sprintf("normal(0, %.4f)", intercept_prior_scale), class = "Intercept")
)

if (parsed$model_type == "categorical") {
  fit <- brms::brm(
    formula = calibration_formula,
    data = calibration_df,
    family = brms::categorical(),
    prior = categorical_priors,
    backend = "rstan",
    chains = chains,
    iter = iter,
    warmup = warmup,
    seed = seed,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth)
  )
  saved_model <- fit
} else {
  misfit_fit <- brms::brm(
    formula = misfit_formula,
    data = calibration_df,
    family = brms::bernoulli(link = "logit"),
    prior = bernoulli_priors,
    backend = "rstan",
    chains = chains,
    iter = iter,
    warmup = warmup,
    seed = seed,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth)
  )
  direction_fit <- brms::brm(
    formula = direction_formula,
    data = direction_df,
    family = brms::bernoulli(link = "logit"),
    prior = bernoulli_priors,
    backend = "rstan",
    chains = chains,
    iter = iter,
    warmup = warmup,
    seed = seed + 1L,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth)
  )
  saved_model <- list(
    model_type = "warning_direction",
    misfit_model = misfit_fit,
    direction_model = direction_fit,
    class_names = c("small", "fit", "large"),
    formula_preset = parsed$formula_preset,
    weighting = parsed$weighting,
    warning_threshold = 0.70
  )
}

dir.create(dirname(parsed$output_model), recursive = TRUE, showWarnings = FALSE)
saveRDS(saved_model, parsed$output_model)

summary_text <- capture.output({
  if (parsed$model_type == "categorical") {
    print(summary(fit))
  } else {
    cat("Misfit model summary:\n")
    print(summary(misfit_fit))
    cat("\nDirection model summary:\n")
    print(summary(direction_fit))
  }
  cat("\nModel type:", parsed$model_type, "\n")
  cat("\nFormula:\n")
  if (parsed$model_type == "categorical") {
    print(calibration_formula)
  } else {
    print(misfit_formula)
    print(direction_formula)
  }
  cat("\nInput rows:", nrow(calibration_df), "\n")
  cat("Max rows:", max_rows, "\n")
  cat("Direction rows:", nrow(direction_df), "\n")
  cat("\nFormula preset:", parsed$formula_preset, "\n")
  cat("Weighting:", parsed$weighting, "\n")
  cat("Class counts:\n")
  print(table(calibration_df$fit_label))
  cat("Class-weight summary:\n")
  print(tapply(calibration_df$calibration_weight, calibration_df$fit_label, summary))
  cat("Misfit-weight summary:\n")
  print(tapply(calibration_df$misfit_weight, calibration_df$misfit_label, summary))
  cat("Direction class counts:\n")
  print(table(direction_df$fit_label))
  cat("\nPrior scale:", prior_scale, "\n")
  cat("Intercept prior scale:", intercept_prior_scale, "\n")
  cat("adapt_delta:", adapt_delta, "\n")
  cat("max_treedepth:", max_treedepth, "\n")
})
writeLines(summary_text, parsed$output_summary)

cat("\nBayesian MLP calibrator trained with brms + rstan.\n")
cat(sprintf("Rows: %d\n", nrow(calibration_df)))
cat(sprintf("Max rows: %d\n", max_rows))
cat(sprintf("Model type: %s\n", parsed$model_type))
cat(sprintf("Formula preset: %s\n", parsed$formula_preset))
cat(sprintf("Weighting: %s\n", parsed$weighting))
cat(sprintf("Model path: %s\n", parsed$output_model))
cat(sprintf("Summary path: %s\n", parsed$output_summary))
