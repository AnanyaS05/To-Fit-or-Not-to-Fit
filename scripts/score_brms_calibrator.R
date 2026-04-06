 # Score candidate DemoFit Co sizes with the saved Bayesian calibrator.
 #
 # This supports both saved model formats:
 # - a single brms categorical model
 # - the default warning-direction bundle with misfit and direction models

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
  model = "artifacts/cold_start/bayesian_mlp_calibrator_rstan.rds",
  input = "artifacts/cold_start/candidate_scores_for_r.csv",
  output = "artifacts/cold_start/calibrated_candidate_scores.json"
)

parsed <- parse_named_args(commandArgs(trailingOnly = TRUE), defaults)

if (!requireNamespace("brms", quietly = TRUE)) {
  stop("Package 'brms' is required. Install it with install.packages('brms').", call. = FALSE)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Package 'jsonlite' is required. Install it with install.packages('jsonlite').", call. = FALSE)
}

model <- readRDS(parsed$model)
candidate_df <- utils::read.csv(parsed$input, stringsAsFactors = FALSE)
candidate_df$source <- factor(candidate_df$source, levels = c("modcloth", "renttherunway"))
candidate_df$garment_type <- factor(candidate_df$garment_type, levels = c("tops", "bottoms", "dresses"))
candidate_df$selected_size_label <- factor(candidate_df$selected_size_label, levels = c("XS", "S", "M", "L", "XL"))
candidate_df$fit_label <- factor("fit", levels = c("small", "fit", "large"))
if (!"calibration_weight" %in% names(candidate_df)) {
  candidate_df$calibration_weight <- 1.0
}
if (!"misfit_weight" %in% names(candidate_df)) {
  candidate_df$misfit_weight <- 1.0
}
if (!"direction_weight" %in% names(candidate_df)) {
  candidate_df$direction_weight <- 1.0
}
candidate_df$misfit_label <- 0L
candidate_df$large_label <- 0L

is_warning_direction <- is.list(model) &&
  !is.null(model$model_type) &&
  identical(model$model_type, "warning_direction")

if (is_warning_direction) {
  # Convert the two Bernoulli posterior means back into the three class probabilities
  # expected by the Python CLI.
  misfit_epred <- brms::posterior_epred(model$misfit_model, newdata = candidate_df)
  direction_epred <- brms::posterior_epred(model$direction_model, newdata = candidate_df)

  p_misfit <- colMeans(misfit_epred)
  p_large_given_misfit <- colMeans(direction_epred)
  mean_probs <- cbind(
    small = p_misfit * (1.0 - p_large_given_misfit),
    fit = 1.0 - p_misfit,
    large = p_misfit * p_large_given_misfit
  )
} else {
  epred <- brms::posterior_epred(model, newdata = candidate_df)
  category_names <- dimnames(epred)[[3]]
  if (is.null(category_names)) {
    category_names <- c("small", "fit", "large")
  }
  category_names <- sub("^mu", "", category_names)
  category_names <- sub("^P\\.", "", category_names)

  mean_probs <- apply(epred, c(2, 3), mean)
  colnames(mean_probs) <- category_names
}

result <- cbind(
  candidate_df[c("source", "garment_type", "selected_size_label")],
  as.data.frame(mean_probs, check.names = FALSE)
)

for (class_name in c("small", "fit", "large")) {
  if (!class_name %in% names(result)) {
    result[[class_name]] <- NA_real_
  }
}

output <- list(
  rows = lapply(seq_len(nrow(result)), function(idx) {
    list(
      source = as.character(result$source[[idx]]),
      garment_type = as.character(result$garment_type[[idx]]),
      selected_size_label = as.character(result$selected_size_label[[idx]]),
      probabilities = list(
        small = unname(result$small[[idx]]),
        fit = unname(result$fit[[idx]]),
        large = unname(result$large[[idx]])
      )
    )
  })
)

jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, digits = 8, pretty = TRUE)
cat(sprintf("Saved calibrated candidate scores to %s\n", parsed$output))
