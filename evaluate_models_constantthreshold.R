library(hdf5r)                                  # Provide read/write access to HDF5 files
library(raster)                                 # For raster data handling
library(dismo)                                  # For species distribution modeling
library(pROC)                                   # For ROC/AUC
library(tidyverse)                              # Data manipulation
library(rjson)                                  # JSON parsing
library(yaml)                                   # YAML parsing
library(rJava)                                  # Java interface
set.seed(42)

source("Utils_R.R")                    # Source custom functions

args = commandArgs(trailingOnly = TRUE)         # Capture command-line arguments
if (length(args) == 0) {
  stop("At least one argument must be supplied (input file).n", call. = FALSE)
}

run_id <- "e52c8ac9a3e24c75ac871f63bbdea060"     # Unique ID for this run
exp_id <- "115656750127464383"                  # Experiment ID

# Base directories
dir_base_run_id <- file.path("predicts_maxent", run_id)
dir_run_id_h5 <- file.path(dir_base_run_id, "h5", "all")
dir_run_id_h5_constantthreshold <- file.path(dir_base_run_id, "h5", "binary_constantthreshold")
create_folder(dir_run_id_h5_constantthreshold)
dir_run_id_png_constantthreshold <- file.path(dir_base_run_id, "png", "binary_constantthreshold")
create_folder(dir_run_id_png_constantthreshold)

# Load DeepSDM config and define environment list
DeepSDM_conf_path <- file.path("predicts", run_id, "DeepSDM_conf.yaml")
DeepSDM_conf <- yaml.load_file(DeepSDM_conf_path)
env_list <- sort(DeepSDM_conf$training_conf$env_list)

# Raster masks and partitions
extent_binary <- raster(DeepSDM_conf$geo_extent_file)
trainval_split <- raster(file.path("mlruns", exp_id, run_id, "artifacts", "extent_binary", "partition_extent.tif"))
i_extent <- which(values(extent_binary) == 1)
i_trainsplit <- which(values(trainval_split) == 1)
i_valsplit <- which(is.na(values(trainval_split)))

# Environmental and species info
env_info_path <- file.path("predicts", run_id, "env_inf.json")
env_info <- fromJSON(file = env_info_path)
sp_info_path <- file.path("predicts", run_id, "sp_inf.json")
sp_info <- fromJSON(file = sp_info_path)

# Date and species lists
date_list_predict <- DeepSDM_conf$training_conf$date_list_predict
date_list_train <- DeepSDM_conf$training_conf$date_list_train
species_list <- sort(DeepSDM_conf$training_conf$species_list_train)

color <- c("#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704")

# Data frame to store threshold-based indicators
df_indi <- data.frame(
  species = character(),
  date = character(),
  maxent_train_TSS = numeric(),
  maxent_train_kappa = numeric(),
  maxent_train_f1 = numeric(),
  deepsdm_train_TSS = numeric(),
  deepsdm_train_kappa = numeric(),
  deepsdm_train_f1 = numeric(),
  maxent_val_TSS = numeric(),
  maxent_val_kappa = numeric(),
  maxent_val_f1 = numeric(),
  deepsdm_val_TSS = numeric(),
  deepsdm_val_kappa = numeric(),
  deepsdm_val_f1 = numeric(),
  train_points = numeric(),
  val_points = numeric(),
  threshold_maxent = numeric(),
  threshold_deepsdm = numeric()
)

# Parse command-line argument for species range
r_start <- as.numeric(args[1])
r_end <- r_start + 2

# Loop over the species subset
for (species in species_list[r_start:min(r_end, length(species_list))]) {
  dir_run_id_h5_sp <- file.path(dir_run_id_h5, species)
  dir_run_id_h5_constantthreshold_sp <- file.path(dir_run_id_h5_constantthreshold, species)
  create_folder(dir_run_id_h5_constantthreshold_sp)
  dir_run_id_png_constantthreshold_sp <- file.path(dir_run_id_png_constantthreshold, species)
  create_folder(dir_run_id_png_constantthreshold_sp)

  # Check existence of HDF5 predictions
  maxent_h5_exists <- FALSE
  maxent_h5_path <- file.path(dir_run_id_h5_sp, sprintf("%s.h5", species))
  maxent_h5_exists <- file.exists(maxent_h5_path)

  deepsdm_h5_exists <- FALSE
  deepsdm_h5_path <- file.path("predicts", run_id, "h5", species, sprintf("%s.h5", species))
  deepsdm_h5_exists <- file.exists(deepsdm_h5_path)

  maxent_10 <- list()
  deepsdm_10 <- list()
  actual_10 <- list()

  # Loop through each date to extract predictions
  for (date in date_list_predict) {
    print(paste("start", species, date))
    set_default_variable()
    generate_points(num_pa = "num_p")

    # Skip if presence/absence points are insufficient
    if (
      nrow(xy_p_month_trainsplit) == 0 ||
      nrow(xy_p_month_valsplit) == 0 ||
      nrow(xy_pa_month_sample_trainsplit) == 0 ||
      nrow(xy_pa_month_sample_valsplit) == 0
    ) {
      next
    }

    # Store actual presence/absence for train/val
    actual_10[[date]] <- list()
    actual_10[[date]]$train_actual_1 <- rep(1, nrow(xy_p_month_trainsplit))
    actual_10[[date]]$train_actual_0 <- rep(0, nrow(xy_pa_month_sample_trainsplit))
    actual_10[[date]]$val_actual_1 <- rep(1, nrow(xy_p_month_valsplit))
    actual_10[[date]]$val_actual_0 <- rep(0, nrow(xy_pa_month_sample_valsplit))
    actual_10[[date]]$xy_p_month <- xy_p_month

    # If Maxent HDF5 exists, load predictions for this date
    maxent_h5_date_exists <- FALSE
    if (maxent_h5_exists) {
      set_default_variable()
      maxent_h5_date_exists <- check_dataset_in_h5(maxent_h5_path, date)
      if (maxent_h5_date_exists) {
        maxent_all_month <- h5dataset_to_raster(maxent_h5_path, date)
        maxent_10[[date]] <- list()
        maxent_10[[date]]$train_pred_1 <- raster::extract(maxent_all_month, xy_p_month_trainsplit) %>% as.numeric()
        maxent_10[[date]]$train_pred_0 <- raster::extract(maxent_all_month, xy_pa_month_sample_trainsplit)
        maxent_10[[date]]$val_pred_1 <- raster::extract(maxent_all_month, xy_p_month_valsplit)
        maxent_10[[date]]$val_pred_0 <- raster::extract(maxent_all_month, xy_pa_month_sample_valsplit)
      }
    }

    # If DeepSDM HDF5 exists, load predictions for this date
    deepsdm_h5_date_exists <- FALSE
    if (deepsdm_h5_exists) {
      set_default_variable()
      deepsdm_h5_date_exists <- check_dataset_in_h5(deepsdm_h5_path, date)
      if (deepsdm_h5_date_exists) {
        deepsdm_all_month <- h5dataset_to_raster(deepsdm_h5_path, date)
        deepsdm_10[[date]] <- list()
        deepsdm_10[[date]]$train_pred_1 <- raster::extract(deepsdm_all_month, xy_p_month_trainsplit) %>% as.numeric()
        deepsdm_10[[date]]$train_pred_0 <- raster::extract(deepsdm_all_month, xy_pa_month_sample_trainsplit)
        deepsdm_10[[date]]$val_pred_1 <- raster::extract(deepsdm_all_month, xy_p_month_valsplit)
        deepsdm_10[[date]]$val_pred_0 <- raster::extract(deepsdm_all_month, xy_pa_month_sample_valsplit)
      }
    }
  }

  # If no valid date produced predictions, skip
  if (length(actual_10) == 0) {
    next
  }

  # Consolidate training predictions across all dates to derive a single threshold
  all_train_maxent_1 <- unlist(lapply(maxent_10, function(sublist) sublist$train_pred_1))
  all_train_maxent_0 <- unlist(lapply(maxent_10, function(sublist) sublist$train_pred_0))
  all_train_deepsdm_1 <- unlist(lapply(deepsdm_10, function(sublist) sublist$train_pred_1))
  all_train_deepsdm_0 <- unlist(lapply(deepsdm_10, function(sublist) sublist$train_pred_0))
  all_train_actual_1 <- unlist(lapply(actual_10, function(sublist) sublist$train_actual_1))
  all_train_actual_0 <- unlist(lapply(actual_10, function(sublist) sublist$train_actual_0))

  # Consolidate validation predictions if needed
  all_val_maxent_1 <- unlist(lapply(maxent_10, function(sublist) sublist$val_pred_1))
  all_val_maxent_0 <- unlist(lapply(maxent_10, function(sublist) sublist$val_pred_0))
  all_val_deepsdm_1 <- unlist(lapply(deepsdm_10, function(sublist) sublist$val_pred_1))
  all_val_deepsdm_0 <- unlist(lapply(deepsdm_10, function(sublist) sublist$val_pred_0))
  all_val_actual_1 <- unlist(lapply(actual_10, function(sublist) sublist$val_actual_1))
  all_val_actual_0 <- unlist(lapply(actual_10, function(sublist) sublist$val_actual_0))

  # Compute ROC for the training data and get "best" threshold
  roc_train_maxent <- roc(c(all_train_actual_1, all_train_actual_0), c(all_train_maxent_1, all_train_maxent_0))
  roc_train_deepsdm <- roc(c(all_train_actual_1, all_train_actual_0), c(all_train_deepsdm_1, all_train_deepsdm_0))
  best_threshold_train_maxent <- coords(roc_train_maxent, "best", ret = c("threshold")) %>% pull() %>% min()
  best_threshold_train_deepsdm <- coords(roc_train_deepsdm, "best", ret = c("threshold")) %>% pull() %>% min()

  # Apply the derived thresholds to each date's continuous predictions, then calculate indicators
  for (date in date_list_predict) {
    if (maxent_h5_exists) {
      if (check_dataset_in_h5(maxent_h5_path, date)) {
        maxent_all_month <- h5dataset_to_raster(maxent_h5_path, date)
        log_binary(
          dir_run_id_h5_constantthreshold,
          dir_run_id_png_constantthreshold,
          species,
          date,
          maxent_all_month,
          best_threshold_train_maxent,
          extent_binary,
          "maxent_all_all_binary_constantthreshold",
          run_id,
          actual_10[[date]]$xy_p_month,
          sprintf("%s_maxent.h5", species)
        )
      }
    }

    if (deepsdm_h5_exists) {
      if (check_dataset_in_h5(deepsdm_h5_path, date)) {
        deepsdm_all_month <- h5dataset_to_raster(deepsdm_h5_path, date)
        log_binary(
          dir_run_id_h5_constantthreshold,
          dir_run_id_png_constantthreshold,
          species,
          date,
          deepsdm_all_month,
          best_threshold_train_deepsdm,
          extent_binary,
          "deepsdm_all_all_binary_constantthreshold",
          run_id,
          actual_10[[date]]$xy_p_month,
          sprintf("%s_deepsdm.h5", species)
        )
      }
    }

    # Calculate threshold-dependent indicators for training
    maxent_train_indi <- calculate_thresholddepend_indi(
      maxent_10[[date]]$train_pred_1,
      maxent_10[[date]]$train_pred_0,
      actual_10[[date]]$train_actual_1,
      actual_10[[date]]$train_actual_0,
      best_threshold_train_maxent
    )
    deepsdm_train_indi <- calculate_thresholddepend_indi(
      deepsdm_10[[date]]$train_pred_1,
      deepsdm_10[[date]]$train_pred_0,
      actual_10[[date]]$train_actual_1,
      actual_10[[date]]$train_actual_0,
      best_threshold_train_deepsdm
    )

    # Calculate threshold-dependent indicators for validation
    maxent_val_indi <- calculate_thresholddepend_indi(
      maxent_10[[date]]$val_pred_1,
      maxent_10[[date]]$val_pred_0,
      actual_10[[date]]$val_actual_1,
      actual_10[[date]]$val_actual_0,
      best_threshold_train_maxent
    )
    deepsdm_val_indi <- calculate_thresholddepend_indi(
      deepsdm_10[[date]]$val_pred_1,
      deepsdm_10[[date]]$val_pred_0,
      actual_10[[date]]$val_actual_1,
      actual_10[[date]]$val_actual_0,
      best_threshold_train_deepsdm
    )

    # Insert row of computed metrics into df_indi
    df_indi[nrow(df_indi) + 1, ] <- c(
      species,
      date,
      unlist(maxent_train_indi),
      unlist(deepsdm_train_indi),
      unlist(maxent_val_indi),
      unlist(deepsdm_val_indi),
      length(actual_10[[date]]$train_actual_1),
      length(actual_10[[date]]$train_actual_0),
      best_threshold_train_maxent,
      best_threshold_train_deepsdm
    )
  }
}

# Write results to CSV
output_path <- file.path(dir_base_run_id, sprintf("model_performance_constantthreshold%03d.csv", r_start))
write.csv(df_indi, output_path, row.names = FALSE)