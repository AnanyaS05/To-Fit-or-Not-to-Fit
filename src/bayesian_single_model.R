# Bayesian Categorical Logistic Regression (brms)
# Train on Data/train_new.csv and test on Data/test_new.csv
# Main-effects model only

set.seed(42)

# -------------------------------
# Configuration
# -------------------------------
train_path <- "C:/Users/anany/Desktop/Northeastern/Year4/Spring2026/To-Fit-or-Not-to-Fit/Data/train_new.csv"
test_path <- "C:/Users/anany/Desktop/Northeastern/Year4/Spring2026/To-Fit-or-Not-to-Fit/Data/test_new.csv"
target_col <- "fit"

# Keep only these model columns (after name sanitization)
model_cols <- c("fit", "size", "cup_size", "hips", "bra_size", "category", "height")
categorical_cols <- c("size", "cup_size", "bra_size", "category")
numeric_cols <- c("hips", "height")

# brms/rstan controls
chains <- 2
requested_cores <- 2
is_rstudio <- identical(Sys.getenv("RSTUDIO"), "1")
has_rstudioapi <- requireNamespace("rstudioapi", quietly = TRUE)
if (is_rstudio && !has_rstudioapi) {
  cores <- 1
  message("rstudioapi not detected in RStudio; falling back to cores = 1 for rstan.")
} else {
  cores <- min(requested_cores, max(1L, parallel::detectCores(logical = FALSE)))
}
iter <- 500
warmup <- 250
adapt_delta <- 0.95
max_treedepth <- 12

# -------------------------------
# Package loading
# -------------------------------
required_pkgs <- c("brms", "rstan", "dplyr", "tibble")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    "Missing packages: ", paste(missing_pkgs, collapse = ", "),
    "\nInstall with: install.packages(c(",
    paste(sprintf("\"%s\"", missing_pkgs), collapse = ", "),
    "))"
  )
}

library(brms)
library(rstan)
library(dplyr)
library(tibble)

rstan_options(auto_write = TRUE)
options(mc.cores = cores)

# -------------------------------
# Helpers
# -------------------------------
sanitize_name <- function(x) {
  x <- trimws(x)
  x <- gsub("[^A-Za-z0-9_]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  x
}

predict_class_from_brms <- function(fit, newdata, class_levels) {
  prob_draws <- posterior_epred(fit, newdata = newdata, re_formula = NA)
  mean_probs <- apply(prob_draws, c(2, 3), mean)
  pred_idx <- max.col(mean_probs, ties.method = "first")
  factor(class_levels[pred_idx], levels = class_levels)
}

compute_metrics <- function(y_true, y_pred, class_levels) {
  y_true <- factor(y_true, levels = class_levels)
  y_pred <- factor(y_pred, levels = class_levels)
  
  cm <- table(y_true, y_pred)
  total <- sum(cm)
  accuracy <- sum(diag(cm)) / total
  
  per_class <- lapply(class_levels, function(cls) {
    tp <- cm[cls, cls]
    fp <- sum(cm[, cls]) - tp
    fn <- sum(cm[cls, ]) - tp
    support <- sum(cm[cls, ])
    
    precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    recall <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f1 <- if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0
    
    tibble(
      class = cls,
      precision = as.numeric(precision),
      recall = as.numeric(recall),
      f1 = as.numeric(f1),
      support = as.integer(support)
    )
  }) %>% bind_rows()
  
  summary <- tibble(
    accuracy = as.numeric(accuracy),
    macro_precision = mean(per_class$precision),
    macro_recall = mean(per_class$recall),
    macro_f1 = mean(per_class$f1)
  )
  
  list(confusion_matrix = cm, per_class = per_class, summary = summary)
}

# -------------------------------
# Load and sanitize data
# -------------------------------
train_df <- read.csv(train_path, stringsAsFactors = FALSE, check.names = FALSE)
test_df <- read.csv(test_path, stringsAsFactors = FALSE, check.names = FALSE)

colnames(train_df) <- sanitize_name(colnames(train_df))
colnames(test_df) <- sanitize_name(colnames(test_df))

model_cols <- sanitize_name(model_cols)
categorical_cols <- sanitize_name(categorical_cols)
numeric_cols <- sanitize_name(numeric_cols)
target_col <- sanitize_name(target_col)

missing_train_cols <- setdiff(model_cols, colnames(train_df))
missing_test_cols <- setdiff(model_cols, colnames(test_df))
if (length(missing_train_cols) > 0) {
  stop("Missing required model columns in train: ", paste(missing_train_cols, collapse = ", "))
}
if (length(missing_test_cols) > 0) {
  stop("Missing required model columns in test: ", paste(missing_test_cols, collapse = ", "))
}

# Subset to only model columns before preprocessing
train_df <- train_df[, model_cols, drop = FALSE]
test_df  <- test_df[, model_cols, drop = FALSE]

# Convert categorical predictors to factors using train levels; unseen/blank test -> UNK
for (col in categorical_cols) {
  train_chr <- trimws(as.character(train_df[[col]]))
  test_chr  <- trimws(as.character(test_df[[col]]))
  
  train_chr[is.na(train_chr) | train_chr == ""] <- "UNK"
  train_levels <- sort(unique(train_chr))
  if (!("UNK" %in% train_levels)) {
    train_levels <- c(train_levels, "UNK")
  }
  
  test_chr[is.na(test_chr) | test_chr == ""] <- "UNK"
  test_chr[!(test_chr %in% train_levels)] <- "UNK"
  
  train_df[[col]] <- factor(train_chr, levels = train_levels)
  test_df[[col]]  <- factor(test_chr, levels = train_levels)
}

# Prepare target as string first so missing/blank can be handled cleanly
train_y_chr <- trimws(as.character(train_df[[target_col]]))
test_y_chr  <- trimws(as.character(test_df[[target_col]]))
train_y_chr[train_y_chr == ""] <- NA
test_y_chr[test_y_chr == ""] <- NA
train_df[[target_col]] <- train_y_chr
test_df[[target_col]]  <- test_y_chr

# Drop rows with missing values in any model variable before fitting
n_train_before <- nrow(train_df)
n_test_before <- nrow(test_df)
train_df <- train_df[stats::complete.cases(train_df[, model_cols]), , drop = FALSE]
test_df  <- test_df[stats::complete.cases(test_df[, model_cols]), , drop = FALSE]

# Target factor levels set from train and applied to test
train_df[[target_col]] <- factor(as.character(train_df[[target_col]]))
class_levels <- levels(train_df[[target_col]])
test_df[[target_col]] <- factor(as.character(test_df[[target_col]]), levels = class_levels)

# If any unseen target labels appear in test, drop those rows
test_df <- test_df[!is.na(test_df[[target_col]]), , drop = FALSE]

# Scale only numeric predictors using train mean/sd, then apply to test
scaler <- list()
for (col in numeric_cols) {
  train_df[[col]] <- as.numeric(train_df[[col]])
  test_df[[col]]  <- as.numeric(test_df[[col]])
  
  mu <- mean(train_df[[col]], na.rm = TRUE)
  sdv <- stats::sd(train_df[[col]], na.rm = TRUE)
  if (!is.finite(sdv) || sdv == 0) sdv <- 1
  
  train_df[[col]] <- (train_df[[col]] - mu) / sdv
  test_df[[col]]  <- (test_df[[col]] - mu) / sdv
  
  scaler[[col]] <- list(mean = mu, sd = sdv)
}

cat("Train rows before/after complete-case filtering:", n_train_before, "/", nrow(train_df), "\n")
cat("Test rows before/after complete-case filtering:", n_test_before, "/", nrow(test_df), "\n")
cat("Model columns:", paste(model_cols, collapse = ", "), "\n")
cat("Class levels:", paste(class_levels, collapse = ", "), "\n")

# -------------------------------
# Main-effects model only
# -------------------------------
main_formula <- bf(
  as.formula(
    paste(target_col, "~ size + cup_size + hips + bra_size + category + height")
  )
)

family_obj <- categorical(link = "logit")

priors_main <- default_prior(
  object = main_formula,
  data = train_df,
  family = family_obj
)

cat("\nFitting: main_effects\n")

best_fit <- brm(
  formula = main_formula,
  data = train_df,
  family = family_obj,
  prior = priors_main,
  chains = chains,
  cores = cores,
  iter = iter,
  warmup = warmup,
  seed = 2026,
  control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth),
  refresh = 10
)

# -------------------------------
# Test evaluation
# -------------------------------
test_pred <- predict_class_from_brms(best_fit, test_df, class_levels)
test_metrics <- compute_metrics(test_df[[target_col]], test_pred, class_levels)

cat("\nTest metrics summary:\n")
print(test_metrics$summary)

cat("\nPer-class metrics:\n")
print(test_metrics$per_class)

cat("\nConfusion matrix:\n")
print(test_metrics$confusion_matrix)

# -------------------------------
# Save artifacts
# -------------------------------
test_model_summary <- tibble(
  model = "main_effects",
  formula = "fit ~ size + cup_size + hips + bra_size + category + height"
)

write.csv(test_model_summary, "../Data/brms_model_used.csv", row.names = FALSE)
write.csv(test_metrics$summary, "../Data/brms_test_summary.csv", row.names = FALSE)
write.csv(test_metrics$per_class, "../Data/brms_test_per_class.csv", row.names = FALSE)

cm_df <- as.data.frame.matrix(test_metrics$confusion_matrix)
cm_df <- tibble::rownames_to_column(cm_df, var = "true_class")
write.csv(cm_df, "../Data/brms_test_confusion_matrix.csv", row.names = FALSE)

cat("\nSaved evaluation artifacts to Data/ directory.\n")