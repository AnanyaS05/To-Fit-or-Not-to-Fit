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

# Cross-validation controls
cv_n_folds <- 5
cv_seed <- 42
cv_chains <- 2
cv_requested_cores <- 1
cv_iter <- 800
cv_warmup <- 300
cv_refresh <- 0

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

stratified_kfold_indices <- function(y, k = 5, seed = 42) {
  set.seed(seed)
  y <- as.factor(y)
  fold_bins <- vector("list", k)
  for (i in seq_len(k)) fold_bins[[i]] <- integer(0)

  for (cls in levels(y)) {
    idx <- which(y == cls)
    idx <- sample(idx, length(idx))

    for (fold in seq_len(k)) {
      cls_fold_idx <- idx[seq(fold, length(idx), by = k)]
      fold_bins[[fold]] <- c(fold_bins[[fold]], cls_fold_idx)
    }
  }

  lapply(fold_bins, function(v) sort(unique(v)))
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
# Visualizations: class distribution
# -------------------------------
png(file.path(out_dir, "brms_class_distribution.png"), width = 1200, height = 500)
par(mfrow = c(1, 2), mar = c(5, 4, 4, 1))

train_counts <- table(train_df[[target_col]])
barplot(
  train_counts,
  col = "#4C72B0",
  main = "Train Class Distribution",
  xlab = "fit class",
  ylab = "count"
)

test_counts <- table(test_df[[target_col]])
barplot(
  test_counts,
  col = "#55A868",
  main = "Test Class Distribution",
  xlab = "fit class",
  ylab = "count"
)

dev.off()

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

# -------------------------------
# Stratified k-fold cross-validation
# -------------------------------
cat("\nRunning stratified ", cv_n_folds, "-fold cross-validation...\n", sep = "")

cv_cores <- min(cv_requested_cores, cores)
cv_folds <- stratified_kfold_indices(train_df[[target_col]], k = cv_n_folds, seed = cv_seed)
all_train_idx <- seq_len(nrow(train_df))
cv_rows <- vector("list", length = cv_n_folds)

for (fold in seq_len(cv_n_folds)) {
  val_idx <- cv_folds[[fold]]
  train_idx <- setdiff(all_train_idx, val_idx)

  cv_train <- train_df[train_idx, , drop = FALSE]
  cv_val <- train_df[val_idx, , drop = FALSE]

  priors_cv <- default_prior(
    object = main_formula,
    data = cv_train,
    family = family_obj
  )

  cv_fit <- brm(
    formula = main_formula,
    data = cv_train,
    family = family_obj,
    prior = priors_cv,
    chains = cv_chains,
    cores = cv_cores,
    iter = cv_iter,
    warmup = cv_warmup,
    seed = 3000 + fold,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth),
    refresh = cv_refresh
  )

  cv_pred <- predict_class_from_brms(cv_fit, cv_val, class_levels)
  cv_metrics <- compute_metrics(cv_val[[target_col]], cv_pred, class_levels)

  cv_rows[[fold]] <- tibble(
    fold = fold,
    val_accuracy = cv_metrics$summary$accuracy[[1]],
    val_macro_precision = cv_metrics$summary$macro_precision[[1]],
    val_macro_recall = cv_metrics$summary$macro_recall[[1]],
    val_macro_f1 = cv_metrics$summary$macro_f1[[1]]
  )

  cat(
    sprintf(
      "Fold %d: accuracy=%.4f, macro_f1=%.4f\n",
      fold,
      cv_rows[[fold]]$val_accuracy,
      cv_rows[[fold]]$val_macro_f1
    )
  )
}

cv_results_df <- bind_rows(cv_rows)
cv_summary_df <- cv_results_df %>%
  summarise(
    mean_accuracy = mean(val_accuracy),
    sd_accuracy = sd(val_accuracy),
    mean_macro_precision = mean(val_macro_precision),
    sd_macro_precision = sd(val_macro_precision),
    mean_macro_recall = mean(val_macro_recall),
    sd_macro_recall = sd(val_macro_recall),
    mean_macro_f1 = mean(val_macro_f1),
    sd_macro_f1 = sd(val_macro_f1)
  )

cat("\nCross-validation fold metrics:\n")
print(cv_results_df)

cat("\nCross-validation summary (mean/std):\n")
print(cv_summary_df)

png(file.path(out_dir, "brms_cv_metrics.png"), width = 1000, height = 550)
par(mar = c(5, 5, 4, 2))
plot(
  cv_results_df$fold,
  cv_results_df$val_accuracy,
  type = "b",
  pch = 19,
  col = "#4C72B0",
  ylim = c(0, 1),
  xlab = "Fold",
  ylab = "Score",
  main = "Cross-Validation Metrics by Fold"
)
lines(
  cv_results_df$fold,
  cv_results_df$val_macro_f1,
  type = "b",
  pch = 17,
  col = "#55A868"
)
abline(h = cv_summary_df$mean_accuracy, lty = 2, col = "#4C72B0")
abline(h = cv_summary_df$mean_macro_f1, lty = 2, col = "#55A868")
legend(
  "bottomleft",
  legend = c("Accuracy", "Macro F1"),
  col = c("#4C72B0", "#55A868"),
  pch = c(19, 17),
  lty = 1,
  bty = "n"
)
dev.off()

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
# Visualizations: metrics and confusion matrix
# -------------------------------
summary_vec <- as.numeric(test_metrics$summary[1, c("accuracy", "macro_precision", "macro_recall", "macro_f1")])
names(summary_vec) <- c("accuracy", "macro_precision", "macro_recall", "macro_f1")

png(file.path(out_dir, "brms_test_summary_metrics.png"), width = 900, height = 550)
par(mar = c(6, 4, 4, 1))
bars <- barplot(
  summary_vec,
  col = "#4C72B0",
  ylim = c(0, 1),
  main = "Bayesian Model Test Summary Metrics",
  ylab = "score",
  las = 2
)
text(bars, summary_vec + 0.03, labels = sprintf("%.3f", summary_vec), cex = 0.9)
dev.off()

png(file.path(out_dir, "brms_per_class_f1.png"), width = 900, height = 550)
par(mar = c(6, 4, 4, 1))
f1_vals <- test_metrics$per_class$f1
f1_names <- as.character(test_metrics$per_class$class)
bars <- barplot(
  f1_vals,
  names.arg = f1_names,
  col = "#55A868",
  ylim = c(0, 1),
  main = "Per-Class F1",
  xlab = "class",
  ylab = "f1 score"
)
text(bars, f1_vals + 0.03, labels = sprintf("%.3f", f1_vals), cex = 0.9)
dev.off()

cm_mat <- as.matrix(test_metrics$confusion_matrix)
cm_plot <- cm_mat[nrow(cm_mat):1, , drop = FALSE]

png(file.path(out_dir, "brms_confusion_matrix_heatmap.png"), width = 800, height = 650)
par(mar = c(5, 5, 4, 2))
image(
  x = seq_len(ncol(cm_plot)),
  y = seq_len(nrow(cm_plot)),
  z = t(cm_plot),
  col = colorRampPalette(c("#f7fbff", "#08306b"))(120),
  axes = FALSE,
  xlab = "Predicted class",
  ylab = "True class",
  main = "Confusion Matrix"
)
axis(1, at = seq_len(ncol(cm_plot)), labels = colnames(cm_mat))
axis(2, at = seq_len(nrow(cm_plot)), labels = rev(rownames(cm_mat)))

threshold <- max(cm_plot) * 0.55
for (i in seq_len(nrow(cm_plot))) {
  for (j in seq_len(ncol(cm_plot))) {
    value <- cm_plot[i, j]
    text_col <- ifelse(value > threshold, "white", "black")
    text(j, i, labels = value, col = text_col, cex = 1.0)
  }
}
dev.off()

# -------------------------------
# Save artifacts
# -------------------------------
model_summary <- tibble(
  model = "categorical_numeric_main_effects",
  formula = "fit ~ size + cup_size + hips + bra_size + category + height",
  family = "categorical(logit)"
)

write.csv(model_summary, file.path(out_dir, "brms_model_used.csv"), row.names = FALSE)
write.csv(cv_results_df, file.path(out_dir, "brms_cv_fold_metrics.csv"), row.names = FALSE)
write.csv(cv_summary_df, file.path(out_dir, "brms_cv_summary.csv"), row.names = FALSE)
write.csv(test_metrics$summary, file.path(out_dir, "brms_test_summary.csv"), row.names = FALSE)
write.csv(test_metrics$per_class, file.path(out_dir, "brms_test_per_class.csv"), row.names = FALSE)

cm_df <- as.data.frame.matrix(test_metrics$confusion_matrix)
cm_df <- tibble::rownames_to_column(cm_df, var = "true_class")
write.csv(cm_df, file.path(out_dir, "brms_test_confusion_matrix.csv"), row.names = FALSE)

cat("\nSaved evaluation artifacts to:", out_dir, "\n")