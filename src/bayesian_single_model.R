# Faster Bayesian categorical logistic regression (brms)
# Keeps categorical family, but uses numeric-coded predictors
# and a simpler, more stable preprocessing pipeline

set.seed(42)

# -------------------------------
# Configuration
# -------------------------------
train_path <- "C:/Users/anany/Desktop/Northeastern/Year4/Spring2026/To-Fit-or-Not-to-Fit/Data/train_new.csv"
test_path  <- "C:/Users/anany/Desktop/Northeastern/Year4/Spring2026/To-Fit-or-Not-to-Fit/Data/test_new.csv"
out_dir <- dirname(train_path)

target_col <- "fit"
predictor_cols <- c("size", "cup_size", "hips", "bra_size", "category", "height")

chains <- 4
requested_cores <- 2
iter <- 1000
warmup <- 200
adapt_delta <- 0.90
max_treedepth <- 10
refresh <- 10

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

cores <- min(requested_cores, max(1L, parallel::detectCores(logical = FALSE)))
options(mc.cores = cores)
rstan_options(auto_write = TRUE)

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

scale_with_train <- function(train_vec, test_vec) {
  mu <- mean(train_vec, na.rm = TRUE)
  sdv <- sd(train_vec, na.rm = TRUE)
  if (!is.finite(sdv) || sdv == 0) sdv <- 1
  
  list(
    train = (train_vec - mu) / sdv,
    test  = (test_vec - mu) / sdv,
    mean = mu,
    sd = sdv
  )
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
# Load data
# -------------------------------
train_df <- read.csv(train_path, stringsAsFactors = FALSE, check.names = FALSE)
test_df  <- read.csv(test_path, stringsAsFactors = FALSE, check.names = FALSE)

colnames(train_df) <- sanitize_name(colnames(train_df))
colnames(test_df)  <- sanitize_name(colnames(test_df))

target_col <- sanitize_name(target_col)
predictor_cols <- sanitize_name(predictor_cols)
model_cols <- c(target_col, predictor_cols)

missing_train <- setdiff(model_cols, names(train_df))
missing_test  <- setdiff(model_cols, names(test_df))

if (length(missing_train) > 0) {
  stop("Missing required columns in train: ", paste(missing_train, collapse = ", "))
}
if (length(missing_test) > 0) {
  stop("Missing required columns in test: ", paste(missing_test, collapse = ", "))
}

train_df <- train_df[, model_cols, drop = FALSE]
test_df  <- test_df[, model_cols, drop = FALSE]

# -------------------------------
# Type handling
# -------------------------------
# Treat predictors as numeric-coded variables, not factors
for (col in predictor_cols) {
  train_df[[col]] <- as.numeric(train_df[[col]])
  test_df[[col]]  <- as.numeric(test_df[[col]])
}

# Target stays categorical
class_levels <- sort(unique(train_df[[target_col]]))
class_levels <- as.character(class_levels)

train_df[[target_col]] <- factor(as.character(train_df[[target_col]]),
                                 levels = class_levels)

test_df[[target_col]] <- factor(as.character(test_df[[target_col]]),
                                levels = class_levels)

# Drop bad rows just in case
train_df <- train_df[complete.cases(train_df), , drop = FALSE]
test_df  <- test_df[complete.cases(test_df), , drop = FALSE]
test_df  <- test_df[!is.na(test_df[[target_col]]), , drop = FALSE]

# -------------------------------
# Scale numeric predictors
# -------------------------------
scale_cols <- c("size", "cup_size", "hips", "bra_size", "height")
scale_cols <- intersect(scale_cols, predictor_cols)

scaler <- list()
for (col in scale_cols) {
  scaled <- scale_with_train(train_df[[col]], test_df[[col]])
  train_df[[col]] <- scaled$train
  test_df[[col]]  <- scaled$test
  scaler[[col]] <- list(mean = scaled$mean, sd = scaled$sd)
}

# keep category as 0/1 numeric
if ("category" %in% predictor_cols) {
  train_df$category <- as.numeric(train_df$category)
  test_df$category  <- as.numeric(test_df$category)
}

cat("Train rows:", nrow(train_df), "\n")
cat("Test rows:", nrow(test_df), "\n")
cat("Class levels:", paste(class_levels, collapse = ", "), "\n")

# -------------------------------
# Main-effects categorical model
# -------------------------------
main_formula <- bf(
  fit ~ size + cup_size + hips + bra_size + category + height
)

family_obj <- categorical(link = "logit")

# Use brms-generated valid priors for categorical model parameters
priors_main <- default_prior(
  object = main_formula,
  data = train_df,
  family = family_obj
)

cat("\nUsing priors:\n")
print(priors_main)

cat("\nFitting categorical Bayesian model...\n")

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
  refresh = refresh
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
model_summary <- tibble(
  model = "categorical_numeric_main_effects",
  formula = "fit ~ size + cup_size + hips + bra_size + category + height",
  family = "categorical(logit)"
)

write.csv(model_summary, file.path(out_dir, "brms_model_used.csv"), row.names = FALSE)
write.csv(test_metrics$summary, file.path(out_dir, "brms_test_summary.csv"), row.names = FALSE)
write.csv(test_metrics$per_class, file.path(out_dir, "brms_test_per_class.csv"), row.names = FALSE)

cm_df <- as.data.frame.matrix(test_metrics$confusion_matrix)
cm_df <- tibble::rownames_to_column(cm_df, var = "true_class")
write.csv(cm_df, file.path(out_dir, "brms_test_confusion_matrix.csv"), row.names = FALSE)

cat("\nSaved evaluation artifacts to:", out_dir, "\n")