GARMENT_LEVELS <- c("tops", "bottoms", "dresses")
SIZE_LEVELS <- c("XS", "S", "M", "L", "XL")
CONTINUOUS_FEATURES <- c(
  "size_ord",
  "height_gap",
  "bust_gap",
  "waist_gap",
  "hips_gap",
  "abs_height_gap",
  "abs_bust_gap",
  "abs_waist_gap",
  "abs_hips_gap"
)

parse_named_args <- function(args, defaults) {
  parsed <- defaults
  idx <- 1L

  while (idx <= length(args)) {
    key <- args[[idx]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    }

    name <- sub("^--", "", key)
    if (idx == length(args)) {
      stop(sprintf("Missing value for argument: %s", key), call. = FALSE)
    }

    parsed[[name]] <- args[[idx + 1L]]
    idx <- idx + 2L
  }

  parsed
}

build_design_matrix <- function(df, numeric_stats = NULL) {
  df <- as.data.frame(df)
  df$garment_type <- factor(df$garment_type, levels = GARMENT_LEVELS)
  df$size_label <- factor(df$size_label, levels = SIZE_LEVELS, ordered = TRUE)

  if (is.null(numeric_stats)) {
    means <- vapply(df[CONTINUOUS_FEATURES], mean, numeric(1), na.rm = TRUE)
    sds <- vapply(df[CONTINUOUS_FEATURES], sd, numeric(1), na.rm = TRUE)
    sds[is.na(sds) | sds == 0] <- 1
    numeric_stats <- list(means = means, sds = sds)
  }

  for (feature_name in CONTINUOUS_FEATURES) {
    df[[feature_name]] <- (
      df[[feature_name]] - numeric_stats$means[[feature_name]]
    ) / numeric_stats$sds[[feature_name]]
  }

  design_formula <- stats::as.formula(
    "~ garment_type + size_label + size_ord + height_gap + bust_gap + waist_gap + hips_gap + abs_height_gap + abs_bust_gap + abs_waist_gap + abs_hips_gap + height_missing + bust_missing + waist_missing + hips_missing"
  )

  X <- stats::model.matrix(design_formula, data = df)
  y <- if ("is_fit" %in% names(df)) as.integer(df$is_fit) else NULL

  list(
    X = X,
    y = y,
    numeric_stats = numeric_stats,
    feature_names = colnames(X)
  )
}

sample_latent_z <- function(y, linear_predictor) {
  y <- as.integer(y)
  latent <- numeric(length(y))

  positive_idx <- which(y == 1L)
  if (length(positive_idx) > 0) {
    lower_cdf <- stats::pnorm(
      0,
      mean = linear_predictor[positive_idx],
      sd = 1
    )
    u <- lower_cdf + stats::runif(length(positive_idx)) * (1 - lower_cdf)
    u <- pmin(pmax(u, 1e-10), 1 - 1e-10)
    latent[positive_idx] <- stats::qnorm(
      u,
      mean = linear_predictor[positive_idx],
      sd = 1
    )
  }

  negative_idx <- which(y == 0L)
  if (length(negative_idx) > 0) {
    upper_cdf <- stats::pnorm(
      0,
      mean = linear_predictor[negative_idx],
      sd = 1
    )
    u <- stats::runif(length(negative_idx)) * upper_cdf
    u <- pmin(pmax(u, 1e-10), 1 - 1e-10)
    latent[negative_idx] <- stats::qnorm(
      u,
      mean = linear_predictor[negative_idx],
      sd = 1
    )
  }

  latent
}

rmvnorm_chol <- function(mean_vec, chol_cov) {
  as.numeric(mean_vec + t(chol_cov) %*% stats::rnorm(length(mean_vec)))
}

fit_bayesian_probit <- function(
  X,
  y,
  burn_in = 1000L,
  samples = 2000L,
  thin = 2L,
  prior_var = 25,
  progress_every = 500L
) {
  X <- as.matrix(X)
  y <- as.integer(y)
  p <- ncol(X)

  prior_precision <- diag(1 / prior_var, p)
  posterior_cov <- solve(crossprod(X) + prior_precision)
  posterior_chol <- chol(posterior_cov)

  total_iterations <- burn_in + samples * thin
  beta <- rep(0, p)
  beta_draws <- matrix(NA_real_, nrow = samples, ncol = p)
  colnames(beta_draws) <- colnames(X)
  draw_idx <- 1L

  for (iteration in seq_len(total_iterations)) {
    linear_predictor <- as.numeric(X %*% beta)
    latent_z <- sample_latent_z(y, linear_predictor)

    posterior_mean <- posterior_cov %*% crossprod(X, latent_z)
    beta <- rmvnorm_chol(posterior_mean, posterior_chol)

    if (iteration > burn_in && ((iteration - burn_in) %% thin == 0L)) {
      beta_draws[draw_idx, ] <- beta
      draw_idx <- draw_idx + 1L
    }

    if (progress_every > 0L && (iteration %% progress_every == 0L)) {
      cat(sprintf("Iteration %d / %d\n", iteration, total_iterations))
    }
  }

  list(
    beta_draws = beta_draws,
    posterior_mean = colMeans(beta_draws),
    posterior_sd = apply(beta_draws, 2, stats::sd)
  )
}

posterior_fit_summary <- function(X_new, beta_draws) {
  X_new <- as.matrix(X_new)
  linear_predictor <- as.numeric(X_new %*% t(beta_draws))
  fit_probs <- stats::pnorm(linear_predictor)

  c(
    mean = mean(fit_probs),
    lower = unname(stats::quantile(fit_probs, probs = 0.05)),
    upper = unname(stats::quantile(fit_probs, probs = 0.95))
  )
}

create_demo_feature_row <- function(
  brand_chart,
  garment_type,
  size_label,
  height_inches = NA_real_,
  bust = NA_real_,
  waist = NA_real_,
  hips = NA_real_
) {
  chart_row <- brand_chart[
    brand_chart$garment_type == garment_type &
      brand_chart$size_label == size_label,
    ,
    drop = FALSE
  ]

  if (nrow(chart_row) != 1L) {
    stop(
      sprintf(
        "Could not find a unique chart row for garment_type=%s and size_label=%s",
        garment_type,
        size_label
      ),
      call. = FALSE
    )
  }

  centers <- list(
    height = (chart_row$height_min_inches + chart_row$height_max_inches) / 2,
    bust = (chart_row$bust_min + chart_row$bust_max) / 2,
    waist = (chart_row$waist_min + chart_row$waist_max) / 2,
    hips = (chart_row$hips_min + chart_row$hips_max) / 2
  )

  observed_height <- if (is.na(height_inches)) centers$height else height_inches
  observed_bust <- if (is.na(bust)) centers$bust else bust
  observed_waist <- if (is.na(waist)) centers$waist else waist
  observed_hips <- if (is.na(hips)) centers$hips else hips

  data.frame(
    garment_type = garment_type,
    size_label = size_label,
    size_ord = match(size_label, SIZE_LEVELS),
    height_gap = observed_height - centers$height,
    bust_gap = observed_bust - centers$bust,
    waist_gap = observed_waist - centers$waist,
    hips_gap = observed_hips - centers$hips,
    abs_height_gap = abs(observed_height - centers$height),
    abs_bust_gap = abs(observed_bust - centers$bust),
    abs_waist_gap = abs(observed_waist - centers$waist),
    abs_hips_gap = abs(observed_hips - centers$hips),
    height_missing = as.integer(is.na(height_inches)),
    bust_missing = as.integer(is.na(bust)),
    waist_missing = as.integer(is.na(waist)),
    hips_missing = as.integer(is.na(hips)),
    stringsAsFactors = FALSE
  )
}
