#!/usr/bin/env Rscript

# Evaluate MaxEnt, SEAM-SDM, and sjSDM 
# procedure in Utils_R.R::generate_points().
#
# Usage:
#   Rscript evaluate_three_models_constantthreshold.R \
#     <species_start_index> sjsdm_conf.yml
#
# The script uses evaluation design:
#   1. For each species-month, read the existing occurrence raster.
#   2. Use generate_points(num_pa = "num_p").
#   3. Extract all three models at the same presence/background coordinates.
#   4. Pool training-partition predictions over months to estimate one fixed
#      threshold per species and model.
#   5. Apply each fixed threshold to monthly training and validation predictions.

suppressPackageStartupMessages({
  library(hdf5r)
  library(raster)
  library(pROC)
  library(tidyverse)
  library(rjson)
  library(yaml)
})

source("Utils_R.R")
source("Utils_sjSDM.R")
set.seed(42)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop(
    paste0(
      "Usage: Rscript evaluate_three_models_constantthreshold_existing_points.R ",
      "<species_start_index> sjsdm_from_rasters_config.yml"
    ),
    call. = FALSE
  )
}

r_start <- as.integer(args[1L])
config_path <- args[2L]
if (!file.exists(config_path)) stop("Missing configuration: ", config_path)
cfg <- yaml::read_yaml(config_path)

run_id <- as.character(cfg$run_id %||% "de4f516ac0ff4a2b999bad82dcb2b6f0")
exp_id <- as.character(cfg$exp_id %||% "688694454263567151")
model_tag <- as.character(cfg$model_tag %||% "sjsdm_dnn_fullcov_gpu")
output_root <- as.character(
  cfg$output_root %||% file.path("predicts_sjsdm", run_id)
)
sjsdm_model_root <- file.path(output_root, model_tag)

DeepSDM_conf_path <- file.path("predicts", run_id, "DeepSDM_conf.yaml")
DeepSDM_conf <- yaml::read_yaml(DeepSDM_conf_path)
env_list <- sort(DeepSDM_conf$training_conf$env_list)

extent_binary <- raster(DeepSDM_conf$geo_extent_file)
partition <- raster(file.path(
  "mlruns", exp_id, run_id,
  "artifacts", "extent_binary", "partition_extent.tif"
))
i_extent <- which(values(extent_binary) == 1)
i_trainsplit <- intersect(which(values(partition) == 1), i_extent)
i_valsplit <- intersect(which(is.na(values(partition))), i_extent)

# These globals are required by generate_points() in Utils_R.R.
env_info <- fromJSON(file = file.path("predicts", run_id, "env_inf.json"))
sp_info <- fromJSON(file = file.path("predicts", run_id, "sp_inf.json"))
date_list_predict <- DeepSDM_conf$training_conf$date_list_predict
date_list_train <- DeepSDM_conf$training_conf$date_list_train
species_list <- sort(DeepSDM_conf$training_conf$species_list_train)

maxent_h5_root <- file.path("predicts_maxent", run_id, "h5", "all")
seam_h5_root <- file.path("predicts", run_id, "h5")
sjsdm_h5_root <- file.path(sjsdm_model_root, "h5", "all")

evaluation_root <- file.path(sjsdm_model_root, "evaluation_constantthreshold")
dir.create(evaluation_root, recursive = TRUE, showWarnings = FALSE)
binary_h5_root <- file.path(evaluation_root, "h5", "binary")
binary_png_root <- file.path(evaluation_root, "png", "binary")
dir.create(binary_h5_root, recursive = TRUE, showWarnings = FALSE)
dir.create(binary_png_root, recursive = TRUE, showWarnings = FALSE)

color <- c(
  "#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c",
  "#f16913", "#d94801", "#a63603", "#7f2704"
)

clean_binary_groups <- function(pred_1, pred_0) {
  pred_1 <- as.numeric(pred_1)
  pred_0 <- as.numeric(pred_0)
  list(
    pred_1 = pred_1[is.finite(pred_1)],
    pred_0 = pred_0[is.finite(pred_0)]
  )
}

safe_auc_from_groups <- function(pred_1, pred_0) {
  x <- clean_binary_groups(pred_1, pred_0)
  if (length(x$pred_1) == 0L || length(x$pred_0) == 0L) return(NA_real_)
  actual <- c(rep(1, length(x$pred_1)), rep(0, length(x$pred_0)))
  pred <- c(x$pred_1, x$pred_0)
  as.numeric(pROC::roc(actual, pred, quiet = TRUE)$auc)
}

safe_best_threshold <- function(pred_1, pred_0) {
  x <- clean_binary_groups(pred_1, pred_0)
  if (length(x$pred_1) == 0L || length(x$pred_0) == 0L) return(NA_real_)
  actual <- c(rep(1, length(x$pred_1)), rep(0, length(x$pred_0)))
  pred <- c(x$pred_1, x$pred_0)
  roc_object <- pROC::roc(actual, pred, quiet = TRUE)
  threshold <- pROC::coords(
    roc_object,
    x = "best",
    best.method = "youden",
    ret = "threshold",
    transpose = FALSE
  )
  threshold <- as.numeric(unlist(threshold, use.names = FALSE))
  threshold <- threshold[is.finite(threshold)]
  if (length(threshold) == 0L) NA_real_ else min(threshold)
}

safe_threshold_indicators <- function(pred_1, pred_0, threshold) {
  x <- clean_binary_groups(pred_1, pred_0)
  if (!is.finite(threshold) ||
      length(x$pred_1) == 0L || length(x$pred_0) == 0L) {
    return(c(TSS = NA_real_, kappa = NA_real_, f1 = NA_real_))
  }

  result <- calculate_thresholddepend_indi(
    pred_1 = x$pred_1,
    pred_0 = x$pred_0,
    actual_1 = rep(1, length(x$pred_1)),
    actual_0 = rep(0, length(x$pred_0)),
    threshold = threshold
  )
  stats::setNames(as.numeric(result[1:3]), c("TSS", "kappa", "f1"))
}

extract_model_predictions <- function(rst) {
  list(
    train_pred_1 = as.numeric(raster::extract(rst, xy_p_month_trainsplit)),
    train_pred_0 = as.numeric(raster::extract(rst, xy_pa_month_sample_trainsplit)),
    val_pred_1 = as.numeric(raster::extract(rst, xy_p_month_valsplit)),
    val_pred_0 = as.numeric(raster::extract(rst, xy_pa_month_sample_valsplit))
  )
}

pool_field <- function(x, field) {
  unlist(lapply(x, `[[`, field), use.names = FALSE)
}

write_binary_prediction <- function(
    model_name,
    species,
    date,
    rst,
    threshold,
    presence_xy) {

  model_h5_root <- file.path(binary_h5_root, model_name)
  model_png_root <- file.path(binary_png_root, model_name)
  create_folder(model_h5_root)
  create_folder(model_png_root)

  log_binary(
    dir_run_id_h5_binary = model_h5_root,
    dir_run_id_png_binary = model_png_root,
    species = species,
    date = date,
    rst = rst,
    threshold = threshold,
    extent_binary = extent_binary,
    log_info = paste0(model_name, "_constantthreshold"),
    timelog = run_id,
    p = presence_xy,
    file_name = sprintf("%s_%s.h5", species, model_name),
    h5 = TRUE,
    png = TRUE
  )
}

results <- list()
result_index <- 0L
r_end <- min(r_start + 4L, length(species_list))

for (species in species_list[r_start:r_end]) {
  message("Evaluating species: ", species)

  maxent_h5_path <- file.path(maxent_h5_root, species, sprintf("%s.h5", species))
  seam_h5_path <- file.path(seam_h5_root, species, sprintf("%s.h5", species))
  sjsdm_h5_path <- file.path(sjsdm_h5_root, species, sprintf("%s.h5", species))

  if (!all(file.exists(c(maxent_h5_path, seam_h5_path, sjsdm_h5_path)))) {
    warning("Skipping ", species, ": at least one model HDF5 file is missing.")
    next
  }

  maxent_by_date <- list()
  seam_by_date <- list()
  sjsdm_by_date <- list()
  sampling_by_date <- list()

  for (date in date_list_predict) {
    message("  ", date)

    # This is the same point-generation function and pseudo-absence rule used in
    # the existing evaluation scripts. It is called once, before reading models,
    # so all three models are evaluated at exactly the same coordinates.
    set_default_variable()
    point_result <- try(generate_points(num_pa = "num_p"), silent = TRUE)
    if (inherits(point_result, "try-error")) next

    if (
      nrow(xy_p_month_trainsplit) == 0L ||
      nrow(xy_p_month_valsplit) == 0L ||
      nrow(xy_pa_month_sample_trainsplit) == 0L ||
      nrow(xy_pa_month_sample_valsplit) == 0L
    ) {
      next
    }

    if (!(
      check_dataset_in_h5(maxent_h5_path, date) &&
      check_dataset_in_h5(seam_h5_path, date) &&
      check_dataset_in_h5(sjsdm_h5_path, date)
    )) {
      next
    }

    maxent_rst <- h5dataset_to_raster(maxent_h5_path, date, template = extent_binary)
    seam_rst <- h5dataset_to_raster(seam_h5_path, date, template = extent_binary)
    sjsdm_rst <- h5dataset_to_raster(sjsdm_h5_path, date, template = extent_binary)

    maxent_by_date[[date]] <- extract_model_predictions(maxent_rst)
    seam_by_date[[date]] <- extract_model_predictions(seam_rst)
    sjsdm_by_date[[date]] <- extract_model_predictions(sjsdm_rst)
    sampling_by_date[[date]] <- list(
      xy_p_month = xy_p_month,
      n_train_presence = nrow(xy_p_month_trainsplit),
      n_train_background = nrow(xy_pa_month_sample_trainsplit),
      n_val_presence = nrow(xy_p_month_valsplit),
      n_val_background = nrow(xy_pa_month_sample_valsplit)
    )
  }

  valid_dates <- Reduce(
    intersect,
    list(
      names(maxent_by_date), names(seam_by_date),
      names(sjsdm_by_date), names(sampling_by_date)
    )
  )
  if (length(valid_dates) == 0L) next

  maxent_by_date <- maxent_by_date[valid_dates]
  seam_by_date <- seam_by_date[valid_dates]
  sjsdm_by_date <- sjsdm_by_date[valid_dates]
  sampling_by_date <- sampling_by_date[valid_dates]

  threshold_maxent <- safe_best_threshold(
    pool_field(maxent_by_date, "train_pred_1"),
    pool_field(maxent_by_date, "train_pred_0")
  )
  threshold_seam <- safe_best_threshold(
    pool_field(seam_by_date, "train_pred_1"),
    pool_field(seam_by_date, "train_pred_0")
  )
  threshold_sjsdm <- safe_best_threshold(
    pool_field(sjsdm_by_date, "train_pred_1"),
    pool_field(sjsdm_by_date, "train_pred_0")
  )

  for (date in valid_dates) {
    maxent_values <- maxent_by_date[[date]]
    seam_values <- seam_by_date[[date]]
    sjsdm_values <- sjsdm_by_date[[date]]
    sample_info <- sampling_by_date[[date]]

    maxent_train <- safe_threshold_indicators(
      maxent_values$train_pred_1, maxent_values$train_pred_0,
      threshold_maxent
    )
    seam_train <- safe_threshold_indicators(
      seam_values$train_pred_1, seam_values$train_pred_0,
      threshold_seam
    )
    sjsdm_train <- safe_threshold_indicators(
      sjsdm_values$train_pred_1, sjsdm_values$train_pred_0,
      threshold_sjsdm
    )

    maxent_val <- safe_threshold_indicators(
      maxent_values$val_pred_1, maxent_values$val_pred_0,
      threshold_maxent
    )
    seam_val <- safe_threshold_indicators(
      seam_values$val_pred_1, seam_values$val_pred_0,
      threshold_seam
    )
    sjsdm_val <- safe_threshold_indicators(
      sjsdm_values$val_pred_1, sjsdm_values$val_pred_0,
      threshold_sjsdm
    )

    # Preserve the previous binary-map output step for all three models.
    maxent_rst <- h5dataset_to_raster(maxent_h5_path, date, template = extent_binary)
    seam_rst <- h5dataset_to_raster(seam_h5_path, date, template = extent_binary)
    sjsdm_rst <- h5dataset_to_raster(sjsdm_h5_path, date, template = extent_binary)
    write_binary_prediction(
      "maxent", species, date, maxent_rst,
      threshold_maxent, sample_info$xy_p_month
    )
    write_binary_prediction(
      "seam_sdm", species, date, seam_rst,
      threshold_seam, sample_info$xy_p_month
    )
    write_binary_prediction(
      "sjsdm", species, date, sjsdm_rst,
      threshold_sjsdm, sample_info$xy_p_month
    )

    result_index <- result_index + 1L
    results[[result_index]] <- data.frame(
      species = species,
      date = date,

      maxent_train_AUC = safe_auc_from_groups(
        maxent_values$train_pred_1, maxent_values$train_pred_0
      ),
      seam_train_AUC = safe_auc_from_groups(
        seam_values$train_pred_1, seam_values$train_pred_0
      ),
      sjsdm_train_AUC = safe_auc_from_groups(
        sjsdm_values$train_pred_1, sjsdm_values$train_pred_0
      ),
      maxent_val_AUC = safe_auc_from_groups(
        maxent_values$val_pred_1, maxent_values$val_pred_0
      ),
      seam_val_AUC = safe_auc_from_groups(
        seam_values$val_pred_1, seam_values$val_pred_0
      ),
      sjsdm_val_AUC = safe_auc_from_groups(
        sjsdm_values$val_pred_1, sjsdm_values$val_pred_0
      ),

      maxent_train_TSS = maxent_train["TSS"],
      maxent_train_kappa = maxent_train["kappa"],
      maxent_train_f1 = maxent_train["f1"],
      seam_train_TSS = seam_train["TSS"],
      seam_train_kappa = seam_train["kappa"],
      seam_train_f1 = seam_train["f1"],
      sjsdm_train_TSS = sjsdm_train["TSS"],
      sjsdm_train_kappa = sjsdm_train["kappa"],
      sjsdm_train_f1 = sjsdm_train["f1"],

      maxent_val_TSS = maxent_val["TSS"],
      maxent_val_kappa = maxent_val["kappa"],
      maxent_val_f1 = maxent_val["f1"],
      seam_val_TSS = seam_val["TSS"],
      seam_val_kappa = seam_val["kappa"],
      seam_val_f1 = seam_val["f1"],
      sjsdm_val_TSS = sjsdm_val["TSS"],
      sjsdm_val_kappa = sjsdm_val["kappa"],
      sjsdm_val_f1 = sjsdm_val["f1"],

      n_train_presence = sample_info$n_train_presence,
      n_train_background = sample_info$n_train_background,
      n_val_presence = sample_info$n_val_presence,
      n_val_background = sample_info$n_val_background,
      threshold_maxent = threshold_maxent,
      threshold_seam = threshold_seam,
      threshold_sjsdm = threshold_sjsdm,
      n_common_valid_months = length(valid_dates),
      sjsdm_model_tag = model_tag,
      stringsAsFactors = FALSE
    )
  }
}

output <- if (length(results) > 0L) {
  do.call(rbind, results)
} else {
  data.frame()
}

output_path <- file.path(
  evaluation_root,
  sprintf("three_model_constantthreshold_%03d.csv", r_start)
)
write.csv(output, output_path, row.names = FALSE)
message("Saved: ", output_path)
