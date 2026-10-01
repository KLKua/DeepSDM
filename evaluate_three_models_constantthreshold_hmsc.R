#!/usr/bin/env Rscript

# Evaluate MaxEnt, SEAM-SDM, and Hmsc at identical monthly points.
#
# Usage:
#   Rscript evaluate_three_models_constantthreshold.R \
#     <species_start_index> hmsc_conf.yml
#
# One training-derived Youden threshold is estimated per species and model from
# pooled months, then applied unchanged to each held-out validation month.

suppressPackageStartupMessages({
  library(hdf5r)
  library(raster)
  library(pROC)
  library(rjson)
  library(yaml)
})

source("Utils_R.R")
source("Utils_Hmsc.R")
assert_packages(c("hdf5r", "raster", "pROC", "rjson", "yaml"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop(
    "Usage: Rscript evaluate_three_models_constantthreshold.R ",
    "<species_start_index> hmsc_conf.yml",
    call. = FALSE
  )
}

r_start <- suppressWarnings(as.integer(args[1L]))
config_path <- args[2L]
assert_file_exists(config_path, "Hmsc YAML configuration")
cfg <- yaml::read_yaml(config_path)

run_id <- assert_nonempty_character(cfg$run_id, "run_id")
exp_id <- assert_nonempty_character(cfg$exp_id, "exp_id")
model_tag <- assert_nonempty_character(
  cfg$model_tag %||% "hmsc_probit_uniform_background",
  "model_tag"
)
output_root <- as.character(
  cfg$output_root %||% file.path("predicts_hmsc", run_id)
)
hmsc_model_root <- file.path(output_root, model_tag)

DeepSDM_conf_path <- file.path("predicts", run_id, "DeepSDM_conf.yaml")
assert_file_exists(DeepSDM_conf_path, "DeepSDM configuration")
DeepSDM_conf <- yaml::read_yaml(DeepSDM_conf_path)
assert_file_exists(DeepSDM_conf$geo_extent_file, "terrestrial extent raster")
extent_binary <- raster::raster(DeepSDM_conf$geo_extent_file)
partition_path <- file.path(
  "mlruns", exp_id, run_id,
  "artifacts", "extent_binary", "partition_extent.tif"
)
assert_file_exists(partition_path, "partition raster")
partition <- raster::raster(partition_path)
assert_raster_alignment(extent_binary, partition, "partition raster")

i_extent <- which(raster::values(extent_binary) == 1)
i_trainsplit <- intersect(which(raster::values(partition) == 1), i_extent)
i_valsplit <- intersect(which(is.na(raster::values(partition))), i_extent)
if (length(i_extent) == 0L || length(i_trainsplit) == 0L || length(i_valsplit) == 0L) {
  stop("The terrestrial extent, training split, or validation split is empty.",
       call. = FALSE)
}
if (length(intersect(i_trainsplit, i_valsplit)) > 0L) {
  stop("Training and validation raster cells overlap.", call. = FALSE)
}

# Globals required by Utils_R.R::generate_points().
env_info_path <- file.path("predicts", run_id, "env_inf.json")
sp_info_path <- file.path("predicts", run_id, "sp_inf.json")
assert_file_exists(env_info_path, "environment metadata")
assert_file_exists(sp_info_path, "species metadata")
env_info <- rjson::fromJSON(file = env_info_path)
sp_info <- rjson::fromJSON(file = sp_info_path)
date_list_predict <- as.character(unlist(
  DeepSDM_conf$training_conf$date_list_predict,
  use.names = FALSE
))
date_list_train <- as.character(unlist(
  DeepSDM_conf$training_conf$date_list_train,
  use.names = FALSE
))
species_list <- sort(assert_nonempty_character(unlist(
  DeepSDM_conf$training_conf$species_list_train,
  use.names = FALSE
), "species_list_train"))

model_names <- c("maxent", "seam", "hmsc")
model_h5_roots <- c(
  maxent = file.path("predicts_maxent", run_id, "h5", "all"),
  seam = file.path("predicts", run_id, "h5"),
  hmsc = file.path(hmsc_model_root, "h5", "all")
)
missing_roots <- names(model_h5_roots)[!dir.exists(model_h5_roots)]
if (length(missing_roots) > 0L) {
  stop(
    "Missing model prediction root(s): ",
    paste(
      paste0(missing_roots, "=", model_h5_roots[missing_roots]),
      collapse = "; "
    ),
    call. = FALSE
  )
}

evaluation_root <- ensure_dir(file.path(
  hmsc_model_root,
  "evaluation_constantthreshold"
))
binary_h5_root <- ensure_dir(file.path(evaluation_root, "h5", "binary"))
binary_png_root <- ensure_dir(file.path(evaluation_root, "png", "binary"))
write_binary_h5 <- isTRUE(cfg$evaluation$write_binary_h5 %||% TRUE)
write_binary_png <- isTRUE(cfg$evaluation$write_binary_png %||% TRUE)
eval_seed <- as.integer(cfg$evaluation$seed %||% 42L)
batch_size <- assert_positive_integer(
  cfg$evaluation$species_batch_size %||% 5L,
  "evaluation.species_batch_size"
)

# Used by Utils_R.R::log_binary() when PNG output is enabled.
color <- c(
  "#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c",
  "#f16913", "#d94801", "#a63603", "#7f2704"
)

if (length(r_start) != 1L || is.na(r_start) ||
    r_start < 1L || r_start > length(species_list)) {
  stop("species_start_index is outside species_list.", call. = FALSE)
}
r_end <- min(r_start + batch_size - 1L, length(species_list))
results <- list()
result_index <- 0L

for (species in species_list[r_start:r_end]) {
  hmsc_log("Evaluating species: ", species)
  h5_paths <- stats::setNames(
    file.path(model_h5_roots, species, sprintf("%s.h5", species)),
    model_names
  )
  if (!all(file.exists(h5_paths))) {
    warning(
      "Skipping ", species, ": missing model HDF5: ",
      paste(names(h5_paths)[!file.exists(h5_paths)], collapse = ", ")
    )
    next
  }

  predictions_by_model <- stats::setNames(
    lapply(model_names, function(x) list()),
    model_names
  )
  sampling_by_date <- list()

  for (date in date_list_predict) {
    # A stable species-month seed makes pseudo-absence coordinates identical
    # regardless of batch size or parallel job order.
    set.seed(stable_seed(eval_seed, "evaluation", species, date))
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

    datasets_exist <- vapply(
      h5_paths,
      function(path) check_dataset_in_h5(path, date),
      logical(1)
    )
    if (!all(datasets_exist)) next

    for (model_name in model_names) {
      rst <- h5dataset_to_raster(
        h5_paths[[model_name]],
        date,
        template = extent_binary
      )
      predictions_by_model[[model_name]][[date]] <- extract_model_predictions(
        rst = rst,
        train_presence_xy = xy_p_month_trainsplit,
        train_background_xy = xy_pa_month_sample_trainsplit,
        val_presence_xy = xy_p_month_valsplit,
        val_background_xy = xy_pa_month_sample_valsplit
      )
    }

    sampling_by_date[[date]] <- list(
      xy_p_month = xy_p_month,
      n_train_presence = nrow(xy_p_month_trainsplit),
      n_train_background = nrow(xy_pa_month_sample_trainsplit),
      n_val_presence = nrow(xy_p_month_valsplit),
      n_val_background = nrow(xy_pa_month_sample_valsplit)
    )
  }

  valid_dates <- names(sampling_by_date)
  if (length(valid_dates) == 0L) next

  thresholds <- stats::setNames(rep(NA_real_, length(model_names)), model_names)
  for (model_name in model_names) {
    model_dates <- predictions_by_model[[model_name]][valid_dates]
    thresholds[[model_name]] <- safe_best_threshold(
      pool_field(model_dates, "train_pred_1"),
      pool_field(model_dates, "train_pred_0")
    )
  }

  for (date in valid_dates) {
    sample_info <- sampling_by_date[[date]]
    row <- data.frame(
      species = species,
      date = date,
      n_train_presence = sample_info$n_train_presence,
      n_train_background = sample_info$n_train_background,
      n_val_presence = sample_info$n_val_presence,
      n_val_background = sample_info$n_val_background,
      n_common_valid_months = length(valid_dates),
      hmsc_model_tag = model_tag,
      stringsAsFactors = FALSE
    )

    for (model_name in model_names) {
      values <- predictions_by_model[[model_name]][[date]]
      train_metrics <- safe_threshold_indicators(
        values$train_pred_1,
        values$train_pred_0,
        thresholds[[model_name]]
      )
      val_metrics <- safe_threshold_indicators(
        values$val_pred_1,
        values$val_pred_0,
        thresholds[[model_name]]
      )

      row[[paste0(model_name, "_train_AUC")]] <- safe_auc_from_groups(
        values$train_pred_1,
        values$train_pred_0
      )
      row[[paste0(model_name, "_val_AUC")]] <- safe_auc_from_groups(
        values$val_pred_1,
        values$val_pred_0
      )
      for (metric in c("TSS", "kappa", "f1")) {
        row[[paste0(model_name, "_train_", metric)]] <- train_metrics[[metric]]
        row[[paste0(model_name, "_val_", metric)]] <- val_metrics[[metric]]
      }
      row[[paste0("threshold_", model_name)]] <- thresholds[[model_name]]

      if (write_binary_h5 || write_binary_png) {
        rst <- h5dataset_to_raster(
          h5_paths[[model_name]],
          date,
          template = extent_binary
        )
        write_binary_model_prediction(
          model_name = model_name,
          species = species,
          date = date,
          rst = rst,
          threshold = thresholds[[model_name]],
          presence_xy = sample_info$xy_p_month,
          binary_h5_root = binary_h5_root,
          binary_png_root = binary_png_root,
          extent_binary = extent_binary,
          run_id = run_id,
          write_h5 = write_binary_h5,
          write_png = write_binary_png
        )
      }
    }

    result_index <- result_index + 1L
    results[[result_index]] <- row
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
hmsc_log("Saved evaluation batch: ", output_path)
