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

CLASS_LEVELS <- c("small", "fit", "large")
SOURCE_LEVELS <- c("modcloth", "renttherunway")
GARMENT_LEVELS <- c("tops", "bottoms", "dresses")
SIZE_LEVELS <- c("XS", "S", "M", "L", "XL")
NUMERIC_COLUMNS <- c(
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
REQUIRED_COLUMNS <- c(
  "row_id",
  "fit_label",
  NUMERIC_COLUMNS,
  "source",
  "garment_type",
  "selected_size_label"
)

defaults <- list(
  artifact_dir = "artifacts/cold_start",
  output_dir = "",
  chains = "2",
  iter = "2000",
  warmup = "1000",
  seed = "42",
  algorithm = "sampling",
  weighting = "balanced",
  formula_preset = "full",
  decision_mode = "argmax",
  bias_grid_min = "-1.5",
  bias_grid_max = "1.5",
  bias_grid_step = "0.1",
  prior_scale = "1.0",
  intercept_prior_scale = "1.5",
  adapt_delta = "0.97",
  max_treedepth = "12"
)

parsed <- parse_named_args(commandArgs(trailingOnly = TRUE), defaults)

if (!requireNamespace("brms", quietly = TRUE)) {
  stop("Package 'brms' is required. Install it with install.packages('brms').", call. = FALSE)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Package 'jsonlite' is required. Install it with install.packages('jsonlite').", call. = FALSE)
}
if (!requireNamespace("rstan", quietly = TRUE)) {
  stop("Package 'rstan' is required. Install it with install.packages('rstan').", call. = FALSE)
}

valid_weighting <- c("none", "balanced", "sqrt_balanced")
if (!parsed$weighting %in% valid_weighting) {
  stop(
    sprintf("--weighting must be one of: %s", paste(valid_weighting, collapse = ", ")),
    call. = FALSE
  )
}

valid_formula_presets <- c("compact", "full")
if (!parsed$formula_preset %in% valid_formula_presets) {
  stop(
    sprintf("--formula_preset must be one of: %s", paste(valid_formula_presets, collapse = ", ")),
    call. = FALSE
  )
}

valid_algorithms <- c("sampling", "meanfield", "fullrank")
if (!parsed$algorithm %in% valid_algorithms) {
  stop(
    sprintf("--algorithm must be one of: %s", paste(valid_algorithms, collapse = ", ")),
    call. = FALSE
  )
}

valid_decision_modes <- c("argmax", "macro_f1_bias_search")
if (!parsed$decision_mode %in% valid_decision_modes) {
  stop(
    sprintf("--decision_mode must be one of: %s", paste(valid_decision_modes, collapse = ", ")),
    call. = FALSE
  )
}

artifact_dir <- parsed$artifact_dir
output_dir <- if (identical(parsed$output_dir, "")) artifact_dir else parsed$output_dir
train_path <- file.path(artifact_dir, "train_split.csv")
calibration_path <- file.path(artifact_dir, "calibration_split.csv")
test_path <- file.path(artifact_dir, "test_split.csv")
output_model <- file.path(output_dir, "bayesian_categorical_model.rds")
output_summary <- file.path(output_dir, "bayesian_categorical_summary.txt")
output_calibration <- file.path(output_dir, "bayesian_categorical_calibration_predictions.csv")
output_test <- file.path(output_dir, "bayesian_categorical_test_predictions.csv")
output_metrics <- file.path(output_dir, "bayesian_categorical_metrics.json")

for (path in c(train_path, calibration_path, test_path)) {
  if (!file.exists(path)) {
    stop(
      sprintf(
        "Missing split export: %s. Re-run scripts/train_cold_start_mlp.py with the updated artifact format.",
        path
      ),
      call. = FALSE
    )
  }
}

require_columns <- function(frame, name) {
  missing <- setdiff(REQUIRED_COLUMNS, names(frame))
  if (length(missing) > 0L) {
    stop(
      sprintf(
        "%s is missing columns: %s. Re-run scripts/train_cold_start_mlp.py with the updated artifact format.",
        name,
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }
}

build_class_weights <- function(labels, method) {
  if (method == "none") {
    return(rep(1.0, length(labels)))
  }

  counts <- table(labels)
  balanced <- length(labels) / (length(counts) * counts)
  if (method == "sqrt_balanced") {
    balanced <- sqrt(balanced)
  }

  weights <- as.numeric(balanced[as.character(labels)])
  weights / mean(weights)
}

prepare_frame <- function(frame, numeric_stats = NULL) {
  frame <- as.data.frame(frame, stringsAsFactors = FALSE)
  require_columns(frame, "Split export")

  frame$fit_label <- factor(frame$fit_label, levels = CLASS_LEVELS)
  frame$source <- factor(frame$source, levels = SOURCE_LEVELS)
  frame$garment_type <- factor(frame$garment_type, levels = GARMENT_LEVELS)
  frame$selected_size_label <- factor(frame$selected_size_label, levels = SIZE_LEVELS)

  if (any(is.na(frame$fit_label))) {
    stop("Split export contains unexpected fit labels.", call. = FALSE)
  }
  if (any(is.na(frame$source))) {
    stop("Split export contains unexpected source values.", call. = FALSE)
  }
  if (any(is.na(frame$garment_type))) {
    stop("Split export contains unexpected garment types.", call. = FALSE)
  }
  if (any(is.na(frame$selected_size_label))) {
    stop("Split export contains unexpected selected size labels.", call. = FALSE)
  }

  for (column in NUMERIC_COLUMNS) {
    frame[[column]] <- as.numeric(frame[[column]])
  }

  if (is.null(numeric_stats)) {
    means <- vapply(frame[NUMERIC_COLUMNS], mean, numeric(1), na.rm = TRUE)
    sds <- vapply(frame[NUMERIC_COLUMNS], stats::sd, numeric(1), na.rm = TRUE)
    sds[is.na(sds) | sds == 0] <- 1.0
    numeric_stats <- list(means = means, sds = sds)
  }

  for (column in NUMERIC_COLUMNS) {
    frame[[column]] <- (frame[[column]] - numeric_stats$means[[column]]) / numeric_stats$sds[[column]]
  }

  list(frame = frame, numeric_stats = numeric_stats)
}

extract_mean_probabilities <- function(model_fit, frame) {
  epred <- brms::posterior_epred(model_fit, newdata = frame)
  if (length(dim(epred)) != 3L) {
    stop("Expected posterior_epred to return draws x rows x classes.", call. = FALSE)
  }

  probabilities <- apply(epred, c(2, 3), mean)
  if (is.null(dim(probabilities))) {
    probabilities <- matrix(probabilities, nrow = 1L)
  }

  category_names <- dimnames(epred)[[3]]
  if (is.null(category_names)) {
    category_names <- CLASS_LEVELS
  }
  category_names <- sub("^mu", "", category_names)
  category_names <- sub("^P\\.", "", category_names)
  colnames(probabilities) <- category_names

  for (class_name in CLASS_LEVELS) {
    if (!class_name %in% colnames(probabilities)) {
      stop(sprintf("posterior_epred output is missing class '%s'.", class_name), call. = FALSE)
    }
  }

  probabilities[, CLASS_LEVELS, drop = FALSE]
}

confusion_matrix_counts <- function(y_true, y_pred, labels) {
  matrix <- matrix(
    0L,
    nrow = length(labels),
    ncol = length(labels),
    dimnames = list(labels, labels)
  )

  for (idx in seq_along(y_true)) {
    true_label <- as.character(y_true[[idx]])
    pred_label <- as.character(y_pred[[idx]])
    matrix[true_label, pred_label] <- matrix[true_label, pred_label] + 1L
  }

  matrix
}

classification_report <- function(y_true, y_pred, labels) {
  cm <- confusion_matrix_counts(y_true, y_pred, labels)
  rows <- lapply(seq_along(labels), function(idx) {
    label <- labels[[idx]]
    tp <- as.numeric(cm[idx, idx])
    fp <- as.numeric(sum(cm[, idx]) - cm[idx, idx])
    fn <- as.numeric(sum(cm[idx, ]) - cm[idx, idx])
    support <- as.integer(sum(cm[idx, ]))

    precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0.0
    recall <- if ((tp + fn) > 0) tp / (tp + fn) else 0.0
    f1 <- if ((precision + recall) > 0) {
      2.0 * precision * recall / (precision + recall)
    } else {
      0.0
    }

    list(
      label = label,
      precision = precision,
      recall = recall,
      f1 = f1,
      support = support
    )
  })

  list(report = rows, confusion_matrix = cm)
}

log_loss <- function(y_true, probabilities, labels) {
  indices <- cbind(seq_along(y_true), match(as.character(y_true), labels))
  selected <- probabilities[indices]
  -mean(log(pmax(selected, 1e-12)))
}

multiclass_brier <- function(y_true, probabilities, labels) {
  one_hot <- matrix(0.0, nrow = nrow(probabilities), ncol = length(labels))
  one_hot[cbind(seq_along(y_true), match(as.character(y_true), labels))] <- 1.0
  mean(rowSums((probabilities - one_hot) ^ 2))
}

metrics_for_scored_frame <- function(frame) {
  probabilities <- as.matrix(frame[paste0("bayes_p_", CLASS_LEVELS)])
  row_sums <- rowSums(probabilities)
  max_probability_sum_error <- max(abs(row_sums - 1.0))
  if (max_probability_sum_error > 1e-5) {
    stop(
      sprintf("Bayesian probabilities do not sum to 1 within tolerance. max_error=%.8f", max_probability_sum_error),
      call. = FALSE
    )
  }

  predicted <- factor(frame$bayes_pred_class, levels = CLASS_LEVELS)
  truth <- factor(frame$fit_label, levels = CLASS_LEVELS)
  class_report <- classification_report(truth, predicted, CLASS_LEVELS)
  macro_f1 <- mean(vapply(class_report$report, function(row) row$f1, numeric(1)))
  accuracy <- mean(as.character(truth) == as.character(predicted))

  list(
    rows = nrow(frame),
    max_probability_sum_error = max_probability_sum_error,
    accuracy = accuracy,
    macro_f1 = macro_f1,
    log_loss = log_loss(truth, probabilities, CLASS_LEVELS),
    brier = multiclass_brier(truth, probabilities, CLASS_LEVELS),
    per_class = class_report$report,
    confusion_matrix = unname(class_report$confusion_matrix)
  )
}

predict_classes_from_probabilities <- function(probabilities, class_biases) {
  ordered_biases <- as.numeric(class_biases[CLASS_LEVELS])
  adjusted_scores <- log(pmax(probabilities, 1e-12))
  adjusted_scores <- sweep(adjusted_scores, 2L, ordered_biases, `+`)
  CLASS_LEVELS[max.col(adjusted_scores, ties.method = "first")]
}

macro_f1_from_predictions <- function(truth, predicted) {
  truth <- factor(truth, levels = CLASS_LEVELS)
  predicted <- factor(predicted, levels = CLASS_LEVELS)
  report <- classification_report(truth, predicted, CLASS_LEVELS)
  mean(vapply(report$report, function(row) row$f1, numeric(1)))
}

tune_decision_rule <- function(truth, probabilities, mode, bias_grid) {
  default_biases <- setNames(c(0.0, 0.0, 0.0), CLASS_LEVELS)
  baseline_predictions <- predict_classes_from_probabilities(probabilities, default_biases)
  baseline_macro_f1 <- macro_f1_from_predictions(truth, baseline_predictions)

  if (mode == "argmax") {
    return(list(
      mode = mode,
      class_biases = as.list(unname(default_biases)),
      class_biases_by_label = as.list(default_biases),
      calibration_macro_f1 = baseline_macro_f1
    ))
  }

  grid_values <- seq(bias_grid$min, bias_grid$max, by = bias_grid$step)
  best <- list(
    macro_f1 = baseline_macro_f1,
    class_biases = default_biases
  )

  evaluate_grid <- function(values_small, values_large, incumbent) {
    best_local <- incumbent

    for (small_bias in values_small) {
      for (large_bias in values_large) {
        candidate_biases <- setNames(c(small_bias, 0.0, large_bias), CLASS_LEVELS)
        predictions <- predict_classes_from_probabilities(probabilities, candidate_biases)
        candidate_macro_f1 <- macro_f1_from_predictions(truth, predictions)

        if ((candidate_macro_f1 > best_local$macro_f1) ||
            (isTRUE(all.equal(candidate_macro_f1, best_local$macro_f1)) &&
             sum(abs(candidate_biases)) < sum(abs(best_local$class_biases)))) {
          best_local <- list(
            macro_f1 = candidate_macro_f1,
            class_biases = candidate_biases
          )
        }
      }
    }

    best_local
  }

  best <- evaluate_grid(grid_values, grid_values, best)

  fine_step <- bias_grid$step / 5.0
  if (is.finite(fine_step) && fine_step > 0) {
    best_small <- best$class_biases[["small"]]
    best_large <- best$class_biases[["large"]]
    fine_small_values <- seq(
      max(bias_grid$min, best_small - bias_grid$step),
      min(bias_grid$max, best_small + bias_grid$step),
      by = fine_step
    )
    fine_large_values <- seq(
      max(bias_grid$min, best_large - bias_grid$step),
      min(bias_grid$max, best_large + bias_grid$step),
      by = fine_step
    )
    best <- evaluate_grid(fine_small_values, fine_large_values, best)
  }

  list(
    mode = mode,
    class_biases = as.list(unname(best$class_biases[CLASS_LEVELS])),
    class_biases_by_label = as.list(best$class_biases[CLASS_LEVELS]),
    calibration_macro_f1 = best$macro_f1,
    baseline_argmax_macro_f1 = baseline_macro_f1
  )
}

build_scored_frame <- function(raw_frame, probabilities, decision_rule) {
  predicted_labels <- predict_classes_from_probabilities(
    probabilities,
    unlist(decision_rule$class_biases_by_label, use.names = TRUE)
  )

  scored <- raw_frame
  for (class_name in CLASS_LEVELS) {
    scored[[paste0("bayes_p_", class_name)]] <- probabilities[, class_name]
  }
  scored$bayes_pred_class <- predicted_labels
  scored
}

predict_probabilities_for_split <- function(raw_frame, model_bundle) {
  prepared <- prepare_frame(raw_frame, numeric_stats = model_bundle$numeric_stats)$frame
  extract_mean_probabilities(model_bundle$model, prepared)
}

score_split <- function(raw_frame, model_bundle, decision_rule) {
  probabilities <- predict_probabilities_for_split(raw_frame, model_bundle)
  build_scored_frame(raw_frame, probabilities, decision_rule)
}

extract_fit_diagnostics <- function(fit, algorithm, runtime_seconds, max_treedepth) {
  diagnostics <- list(
    runtime_seconds = runtime_seconds,
    approximate_inference = algorithm != "sampling"
  )

  rhat_values <- suppressWarnings(as.numeric(unlist(brms::rhat(fit))))
  rhat_values <- rhat_values[is.finite(rhat_values)]
  diagnostics$max_rhat <- if (length(rhat_values) > 0L) max(rhat_values) else NA_real_

  neff_values <- suppressWarnings(as.numeric(unlist(brms::neff_ratio(fit))))
  neff_values <- neff_values[is.finite(neff_values)]
  diagnostics$min_neff_ratio <- if (length(neff_values) > 0L) min(neff_values) else NA_real_

  if (algorithm == "sampling") {
    sampler_params <- tryCatch(
      rstan::get_sampler_params(fit$fit, inc_warmup = FALSE),
      error = function(...) NULL
    )
    divergent_transitions <- 0L
    max_treedepth_hits <- 0L

    if (!is.null(sampler_params)) {
      for (params in sampler_params) {
        if ("divergent__" %in% colnames(params)) {
          divergent_transitions <- divergent_transitions + sum(params[, "divergent__"])
        }
        if ("treedepth__" %in% colnames(params)) {
          max_treedepth_hits <- max_treedepth_hits + sum(params[, "treedepth__"] >= max_treedepth)
        }
      }
    }

    diagnostics$divergent_transitions <- divergent_transitions
    diagnostics$max_treedepth_hits <- max_treedepth_hits
  }

  diagnostics
}

chains <- as.integer(parsed$chains)
iter <- as.integer(parsed$iter)
warmup <- as.integer(parsed$warmup)
seed <- as.integer(parsed$seed)
prior_scale <- as.numeric(parsed$prior_scale)
intercept_prior_scale <- as.numeric(parsed$intercept_prior_scale)
adapt_delta <- as.numeric(parsed$adapt_delta)
max_treedepth <- as.integer(parsed$max_treedepth)
bias_grid_min <- as.numeric(parsed$bias_grid_min)
bias_grid_max <- as.numeric(parsed$bias_grid_max)
bias_grid_step <- as.numeric(parsed$bias_grid_step)

if (!is.finite(bias_grid_min) || !is.finite(bias_grid_max) || !is.finite(bias_grid_step) ||
    bias_grid_step <= 0 || bias_grid_min > bias_grid_max) {
  stop("Bias grid values must define a valid numeric range with positive step size.", call. = FALSE)
}

options(mc.cores = max(1L, min(chains, parallel::detectCores(logical = FALSE))))
rstan::rstan_options(auto_write = TRUE)

train_raw <- utils::read.csv(train_path, stringsAsFactors = FALSE)
calibration_raw <- utils::read.csv(calibration_path, stringsAsFactors = FALSE)
test_raw <- utils::read.csv(test_path, stringsAsFactors = FALSE)

train_prepared <- prepare_frame(train_raw)
train_df <- train_prepared$frame
numeric_stats <- train_prepared$numeric_stats
train_df$training_weight <- build_class_weights(train_df$fit_label, parsed$weighting)

terms_by_preset <- list(
  compact = c(
    "size_order",
    "height_gap",
    "bust_gap",
    "waist_gap",
    "hips_gap",
    "height_missing",
    "bust_missing",
    "waist_missing",
    "hips_missing",
    "source",
    "garment_type",
    "selected_size_label"
  ),
  full = c(
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
    "source",
    "garment_type",
    "selected_size_label"
  )
)

formula_text <- paste(
  "fit_label | weights(training_weight) ~",
  paste(terms_by_preset[[parsed$formula_preset]], collapse = " + ")
)
model_formula <- stats::as.formula(formula_text)

priors <- c(
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

if (parsed$algorithm == "sampling") {
  fit_start <- Sys.time()
  fit <- brms::brm(
    formula = model_formula,
    data = train_df,
    family = brms::categorical(link = "logit"),
    prior = priors,
    backend = "rstan",
    algorithm = parsed$algorithm,
    chains = chains,
    iter = iter,
    warmup = warmup,
    seed = seed,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth)
  )
} else {
  fit_start <- Sys.time()
  fit <- brms::brm(
    formula = model_formula,
    data = train_df,
    family = brms::categorical(link = "logit"),
    prior = priors,
    backend = "rstan",
    algorithm = parsed$algorithm,
    iter = iter,
    seed = seed
  )
}
runtime_seconds <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))

calibration_probabilities <- predict_probabilities_for_split(
  calibration_raw,
  list(model = fit, numeric_stats = numeric_stats)
)
decision_rule <- tune_decision_rule(
  truth = calibration_raw$fit_label,
  probabilities = calibration_probabilities,
  mode = parsed$decision_mode,
  bias_grid = list(min = bias_grid_min, max = bias_grid_max, step = bias_grid_step)
)
fit_diagnostics <- extract_fit_diagnostics(fit, parsed$algorithm, runtime_seconds, max_treedepth)

model_bundle <- list(
  model = fit,
  numeric_stats = numeric_stats,
  class_levels = CLASS_LEVELS,
  source_levels = SOURCE_LEVELS,
  garment_levels = GARMENT_LEVELS,
  size_levels = SIZE_LEVELS,
  numeric_columns = NUMERIC_COLUMNS,
  formula_preset = parsed$formula_preset,
  algorithm = parsed$algorithm,
  prior_scale = prior_scale,
  intercept_prior_scale = intercept_prior_scale,
  adapt_delta = adapt_delta,
  max_treedepth = max_treedepth,
  weighting = parsed$weighting,
  formula = formula_text,
  decision_rule = decision_rule,
  diagnostics = fit_diagnostics
)

calibration_scored <- build_scored_frame(calibration_raw, calibration_probabilities, decision_rule)
test_scored <- score_split(test_raw, model_bundle, decision_rule)

calibration_metrics <- metrics_for_scored_frame(calibration_scored)
test_metrics <- metrics_for_scored_frame(test_scored)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
saveRDS(model_bundle, output_model)
utils::write.csv(calibration_scored, output_calibration, row.names = FALSE)
utils::write.csv(test_scored, output_test, row.names = FALSE)

summary_text <- capture.output({
  print(summary(fit))
  cat("\nFormula:\n")
  print(model_formula)
  cat("\nFormula preset:", parsed$formula_preset, "\n")
  cat("Algorithm:", parsed$algorithm, "\n")
  cat("\nWeighting:", parsed$weighting, "\n")
  cat("Decision mode:", parsed$decision_mode, "\n")
  cat("Decision biases:\n")
  print(decision_rule$class_biases_by_label)
  cat("Training rows:", nrow(train_df), "\n")
  cat("Calibration rows:", nrow(calibration_scored), "\n")
  cat("Test rows:", nrow(test_scored), "\n")
  cat(sprintf("Runtime seconds: %.2f\n", runtime_seconds))
  cat("Diagnostics:\n")
  print(fit_diagnostics)
  cat("Class counts:\n")
  print(table(train_df$fit_label))
})
writeLines(summary_text, output_summary)

metrics_payload <- list(
  artifact_dir = artifact_dir,
  output_dir = output_dir,
  formula = formula_text,
  formula_preset = parsed$formula_preset,
  algorithm = parsed$algorithm,
  weighting = parsed$weighting,
  chains = chains,
  iter = iter,
  warmup = warmup,
  seed = seed,
  prior_scale = prior_scale,
  intercept_prior_scale = intercept_prior_scale,
  adapt_delta = adapt_delta,
  max_treedepth = max_treedepth,
  decision_mode = parsed$decision_mode,
  decision_rule = decision_rule,
  diagnostics = fit_diagnostics,
  train_rows = nrow(train_df),
  calibration = calibration_metrics,
  test = test_metrics,
  paths = list(
    model = output_model,
    summary = output_summary,
    calibration_predictions = output_calibration,
    test_predictions = output_test
  )
)

jsonlite::write_json(metrics_payload, output_metrics, auto_unbox = TRUE, pretty = TRUE, digits = 8)

cat("\nStandalone Bayesian categorical classifier trained.\n")
cat(sprintf("Train rows: %d\n", nrow(train_df)))
cat(sprintf("Calibration rows: %d\n", nrow(calibration_scored)))
cat(sprintf("Test rows: %d\n", nrow(test_scored)))
cat(sprintf("Model path: %s\n", output_model))
cat(sprintf("Calibration predictions: %s\n", output_calibration))
cat(sprintf("Test predictions: %s\n", output_test))
