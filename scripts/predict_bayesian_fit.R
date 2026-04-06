args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- dirname(normalizePath(script_path))
source(file.path(script_dir, "bayesian_probit_utils.R"))

defaults <- list(
  model = "artifacts/bayesian_probit_model.rds",
  brand_csv = "Data/demo_brand_sizing.csv",
  garment_type = "tops",
  size = "M",
  height = "NA",
  bust = "NA",
  waist = "NA",
  hips = "NA"
)

parsed <- parse_named_args(commandArgs(trailingOnly = TRUE), defaults)

parse_optional_numeric <- function(value) {
  if (is.null(value) || identical(value, "") || identical(toupper(value), "NA")) {
    return(NA_real_)
  }
  as.numeric(value)
}

model_path <- parsed$model
brand_csv <- parsed$brand_csv
garment_type <- parsed$garment_type
size_label <- parsed$size
height_inches <- parse_optional_numeric(parsed$height)
bust <- parse_optional_numeric(parsed$bust)
waist <- parse_optional_numeric(parsed$waist)
hips <- parse_optional_numeric(parsed$hips)

model_payload <- readRDS(model_path)
brand_chart <- utils::read.csv(brand_csv, stringsAsFactors = FALSE)

feature_row <- create_demo_feature_row(
  brand_chart = brand_chart,
  garment_type = garment_type,
  size_label = size_label,
  height_inches = height_inches,
  bust = bust,
  waist = waist,
  hips = hips
)

design <- build_design_matrix(feature_row, numeric_stats = model_payload$numeric_stats)
fit_summary <- posterior_fit_summary(design$X, model_payload$beta_draws)

chart_row <- brand_chart[
  brand_chart$garment_type == garment_type &
    brand_chart$size_label == size_label,
  ,
  drop = FALSE
]

within_chart <- c(
  height = is.na(height_inches) || (
    height_inches >= chart_row$height_min_inches &&
      height_inches <= chart_row$height_max_inches
  ),
  bust = is.na(bust) || (bust >= chart_row$bust_min && bust <= chart_row$bust_max),
  waist = is.na(waist) || (waist >= chart_row$waist_min && waist <= chart_row$waist_max),
  hips = is.na(hips) || (hips >= chart_row$hips_min && hips <= chart_row$hips_max)
)

decision <- if (fit_summary[["mean"]] >= 0.5) "likely_fit" else "likely_misfit"
confidence <- if (
  fit_summary[["lower"]] > 0.5 || fit_summary[["upper"]] < 0.5
) {
  "high"
} else {
  "moderate"
}

cat("\nBayesian fit prediction\n")
cat(sprintf("Garment type: %s\n", garment_type))
cat(sprintf("Selected size: %s\n", size_label))
cat(sprintf("Posterior fit probability: %.4f\n", fit_summary[["mean"]]))
cat(sprintf("90%% credible interval: [%.4f, %.4f]\n", fit_summary[["lower"]], fit_summary[["upper"]]))
cat(sprintf("Decision: %s\n", decision))
cat(sprintf("Confidence: %s\n", confidence))
cat("\nWithin demo size chart:\n")
cat(sprintf("  height: %s\n", if (within_chart[["height"]]) "yes" else "no"))
cat(sprintf("  bust: %s\n", if (within_chart[["bust"]]) "yes" else "no"))
cat(sprintf("  waist: %s\n", if (within_chart[["waist"]]) "yes" else "no"))
cat(sprintf("  hips: %s\n", if (within_chart[["hips"]]) "yes" else "no"))
