#!/usr/bin/env Rscript

# Fit one joint sjSDM to all focal species
#
# Usage:
#   Rscript run_sjsdm_gpu.R sjsdm_conf.yaml
#
# Key guarantees:
#   - No raw GBIF table is read.
#   - sjSDM() is called once for the final model; all species are columns of Y.
#   - Training rows come from the eta > 0 target-group recording footprint,
#     reconstructed from all available bird-species occurrence rasters.
#   - Any optional downsampling is uniform within that footprint and is performed
#     before the focal-species response matrix is constructed.
#   - Only cells in partition_extent.tif with value == 1 enter model fitting.
#   - Held-out cells (NA in partition_extent.tif) never enter model fitting.
#   - Environmental preprocessing is exactly load_env_month() from Utils_R.R.
#   - The model is GPU-only; the script stops if CUDA is unavailable.
#   - Final predictions cover all terrestrial cells and are written as one HDF5
#     file per species with one dataset per month.

suppressPackageStartupMessages({
  library(raster)
  library(hdf5r)
  library(yaml)
  library(rjson)
  library(reticulate)
  library(pROC)
})

source("Utils_R.R")
source("Utils_sjSDM.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  stop(
    "Usage: Rscript run_sjsdm_gpu.R sjsdm_conf.yaml",
    call. = FALSE
  )
}

config_path <- args[1L]
assert_file_exists(config_path, "sjSDM YAML configuration")
cfg <- yaml::read_yaml(config_path)

run_id <- as.character(cfg$run_id %||% "de4f516ac0ff4a2b999bad82dcb2b6f0")
exp_id <- as.character(cfg$exp_id %||% "688694454263567151")
model_tag <- as.character(cfg$model_tag %||% "sjsdm_dnn_fullcov_gpu")

reticulate_python <- as.character(
  cfg$reticulate_python %||% Sys.getenv("RETICULATE_PYTHON", unset = "")
)
if (nzchar(reticulate_python)) {
  reticulate::use_python(reticulate_python, required = TRUE)
}

# Load sjSDM only after selecting the Python interpreter.
suppressPackageStartupMessages(library(sjSDM))
gpu <- assert_sjsdm_gpu()

# -----------------------------------------------------------------------------
# 1. Load the same project configuration and data locations as the current R code
# -----------------------------------------------------------------------------
DeepSDM_conf_path <- file.path("predicts", run_id, "DeepSDM_conf.yaml")
assert_file_exists(DeepSDM_conf_path, "DeepSDM configuration")
DeepSDM_conf <- yaml::read_yaml(DeepSDM_conf_path)

env_list <- sort(as.character(unlist(
  DeepSDM_conf$training_conf$env_list,
  use.names = FALSE
)))
date_list_train <- as.character(unlist(
  DeepSDM_conf$training_conf$date_list_train,
  use.names = FALSE
))
date_list_predict <- as.character(unlist(
  DeepSDM_conf$training_conf$date_list_predict,
  use.names = FALSE
))
species_list_train <- sort(as.character(unlist(
  DeepSDM_conf$training_conf$species_list_train,
  use.names = FALSE
)))
species_list_predict <- sort(as.character(unlist(
  DeepSDM_conf$training_conf$species_list_predict,
  use.names = FALSE
)))

if (!identical(species_list_train, species_list_predict)) {
  stop(
    paste0(
      "species_list_train and species_list_predict differ. A single joint model ",
      "requires a fixed species column order."
    ),
    call. = FALSE
  )
}
species_list <- species_list_train

extent_binary <- raster::raster(DeepSDM_conf$geo_extent_file)
partition_path <- file.path(
  "mlruns", exp_id, run_id,
  "artifacts", "extent_binary", "partition_extent.tif"
)
assert_file_exists(partition_path, "partition raster")
partition <- raster::raster(partition_path)

if (!raster::compareRaster(
  extent_binary, partition,
  extent = TRUE, rowcol = TRUE, crs = TRUE, res = TRUE,
  stopiffalse = FALSE
)) {
  stop("extent_binary and partition_extent.tif are not aligned.", call. = FALSE)
}

i_extent <- which(raster::values(extent_binary) == 1)
i_trainsplit <- intersect(which(raster::values(partition) == 1), i_extent)
i_valsplit <- intersect(which(is.na(raster::values(partition))), i_extent)

if (length(i_trainsplit) == 0L || length(i_valsplit) == 0L) {
  stop("Training or validation partition contains no terrestrial cells.", call. = FALSE)
}
if (length(intersect(i_trainsplit, i_valsplit)) > 0L) {
  stop("Training and validation partitions overlap.", call. = FALSE)
}

sjsdm_log(
  "Partition cells: train=", length(i_trainsplit),
  ", validation=", length(i_valsplit),
  ", all terrestrial=", length(i_extent)
)

env_info_path <- file.path("predicts", run_id, "env_inf.json")
sp_info_path <- file.path("predicts", run_id, "sp_inf.json")
assert_file_exists(env_info_path, "environment metadata")
assert_file_exists(sp_info_path, "species metadata")
env_info <- rjson::fromJSON(file = env_info_path)
sp_info <- rjson::fromJSON(file = sp_info_path)

# -----------------------------------------------------------------------------
# 2. Output directories
# -----------------------------------------------------------------------------
output_root <- as.character(
  cfg$output_root %||% file.path("predicts_sjsdm", run_id)
)
model_root <- ensure_dir(file.path(output_root, model_tag))
data_dir <- ensure_dir(file.path(model_root, "data"))
model_dir <- ensure_dir(file.path(model_root, "model"))
h5_root <- ensure_dir(file.path(model_root, "h5", "all"))
log_dir <- ensure_dir(file.path(model_root, "logs"))

training_rds <- file.path(data_dir, "joint_training_target_group_eta_gt_zero.rds")
rebuild_training_data <- isTRUE(cfg$rebuild_training_data %||% FALSE)

# -----------------------------------------------------------------------------
# 3. Existing occurrence rasters -> one N x species joint training matrix
# -----------------------------------------------------------------------------
if (rebuild_training_data || !file.exists(training_rds)) {
  configured_footprint_species <- as.character(unlist(
    cfg$training_data$footprint_species_list,
    use.names = FALSE
  ))
  if (length(configured_footprint_species) == 0L) {
    configured_footprint_species <- sort(names(sp_info$file_name))
  }

  training_data <- build_sjsdm_training_from_occurrence_rasters(
    species_list = species_list,
    date_list = date_list_train,
    partition_cells = i_trainsplit,
    max_sampling_rows_total = as.integer(
      cfg$training_data$max_sampling_rows_total %||% 0L
    ),
    footprint_species_list = configured_footprint_species,
    sp_info = sp_info,
    extent_binary = extent_binary,
    env_list = env_list,
    env_info = env_info,
    DeepSDM_conf = DeepSDM_conf,
    seed = as.integer(cfg$training_data$seed %||% 42L)
  )
  saveRDS(training_data, training_rds)
} else {
  sjsdm_log("Loading cached joint training matrix: ", training_rds)
  training_data <- readRDS(training_rds)
}

X_train <- as.data.frame(training_data$X, check.names = FALSE)
Y_train <- as.matrix(training_data$Y)

if (!identical(colnames(Y_train), species_list)) {
  stop("Cached response matrix has a different species order.", call. = FALSE)
}
if (nrow(X_train) != nrow(Y_train)) {
  stop("X_train and Y_train have different row counts.", call. = FALSE)
}
if (any(training_data$metadata$cell %in% i_valsplit)) {
  stop("Validation cells leaked into the sjSDM training matrix.", call. = FALSE)
}

write.csv(
  training_data$metadata,
  file.path(data_dir, "joint_training_row_metadata.csv"),
  row.names = FALSE
)
write.csv(
  training_data$sampling_summary,
  file.path(data_dir, "target_group_sampling_summary.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    species = colnames(Y_train),
    n_presence_rows = colSums(Y_train),
    prevalence_in_sampled_training_rows = colMeans(Y_train)
  ),
  file.path(data_dir, "joint_training_species_summary.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    predictor = env_list,
    model_column = colnames(X_train),
    normalized_by_load_env_month = !env_list %in% as.character(unlist(
      DeepSDM_conf$training_conf$non_normalize_env_list,
      use.names = FALSE
    ))
  ),
  file.path(data_dir, "environment_preprocessing_audit.csv"),
  row.names = FALSE
)

sjsdm_log(
  "Training matrix: ", nrow(Y_train), " sampled grid-month rows x ",
  ncol(Y_train), " species; predictors=", ncol(X_train)
)

# -----------------------------------------------------------------------------
# 4. Fit exactly one final joint model, using training-partition rows only
# -----------------------------------------------------------------------------
fit_start <- Sys.time()
model <- fit_joint_sjsdm_gpu(
  X_train = X_train,
  Y_train = Y_train,
  hidden = as.integer(unlist(cfg$model$hidden %||% c(64L, 64L))),
  activation = as.character(cfg$model$activation %||% "relu"),
  dropout = as.numeric(cfg$model$dropout %||% 0.10),
  lambda_coef = as.numeric(cfg$model$lambda_coef %||% 0.01),
  alpha_coef = as.numeric(cfg$model$alpha_coef %||% 0.50),
  lambda_cov = as.numeric(cfg$model$lambda_cov %||% 0.01),
  alpha_cov = as.numeric(cfg$model$alpha_cov %||% 0.50),
  iter = as.integer(cfg$model$iter %||% 300L),
  step_size = as.integer(cfg$model$step_size %||% 4096L),
  learning_rate = as.numeric(cfg$model$learning_rate %||% 0.01),
  sampling = as.integer(cfg$model$sampling %||% 100L),
  data_loader_cores = as.integer(cfg$model$data_loader_cores %||% 0L),
  scheduler_patience = as.integer(cfg$model$scheduler_patience %||% 20L),
  lr_reduce_factor = as.numeric(cfg$model$lr_reduce_factor %||% 0.50),
  early_stopping_training = as.integer(
    cfg$model$early_stopping_training %||% 50L
  ),
  seed = as.integer(cfg$model$seed %||% 20260302L)
)
fit_elapsed <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))

# Save portable weights and covariance. Prediction continues in this same R
# process because reticulate/PyTorch model serialization is not always portable.
saveRDS(sjSDM::getWeights(model), file.path(model_dir, "model_weights.rds"))
saveRDS(sjSDM::getCov(model), file.path(model_dir, "species_covariance.rds"))
saveRDS(sjSDM::getCor(model), file.path(model_dir, "species_correlation.rds"))
saveRDS(model$history, file.path(model_dir, "training_history.rds"))

fit_record <- list(
  run_id = run_id,
  exp_id = exp_id,
  model_tag = model_tag,
  gpu = gpu$device_name,
  n_training_rows = nrow(Y_train),
  n_species = ncol(Y_train),
  n_predictors = ncol(X_train),
  fit_elapsed_seconds = fit_elapsed,
  species_order = species_list,
  predictor_order = colnames(X_train),
  model_config = cfg$model
)
yaml::write_yaml(fit_record, file.path(model_dir, "fit_record.yml"))

# Training-only smoke test.
smoke_n <- min(20L, nrow(X_train))
smoke <- predict_joint_sjsdm_in_chunks(
  model = model,
  X_new = X_train[seq_len(smoke_n), , drop = FALSE],
  species_order = species_list,
  chunk_size = smoke_n
)
if (any(!is.finite(smoke))) {
  stop("The fitted sjSDM failed the prediction smoke test.", call. = FALSE)
}

# -----------------------------------------------------------------------------
# 5. Apply the fitted model to ALL terrestrial cells for every prediction month
# -----------------------------------------------------------------------------
prediction_chunk_size <- as.integer(cfg$prediction$chunk_size %||% 20000L)
overwrite <- isTRUE(cfg$prediction$overwrite %||% FALSE)
safe_env_names <- training_data$predictor_order

for (date in date_list_predict) {
  sjsdm_log("Predicting all terrestrial cells for ", date)

  env_month <- load_env_month(env_list, env_info, date, DeepSDM_conf)
  X_all <- raster::extract(env_month, i_extent)
  if (is.null(dim(X_all))) X_all <- matrix(X_all, nrow = 1L)
  X_all <- as.matrix(X_all)
  colnames(X_all) <- make.names(env_list, unique = TRUE)
  X_all <- X_all[, safe_env_names, drop = FALSE]

  valid <- stats::complete.cases(X_all)
  prediction_all <- matrix(
    NA_real_,
    nrow = length(i_extent),
    ncol = length(species_list),
    dimnames = list(NULL, species_list)
  )

  if (any(valid)) {
    prediction_all[valid, ] <- predict_joint_sjsdm_in_chunks(
      model = model,
      X_new = X_all[valid, , drop = FALSE],
      species_order = species_list,
      chunk_size = prediction_chunk_size
    )
  }

  write_sjsdm_month_to_species_h5(
    prediction_matrix = prediction_all,
    species_list = species_list,
    land_cells = i_extent,
    extent_binary = extent_binary,
    date = date,
    h5_root = h5_root,
    overwrite = overwrite
  )

  rm(env_month, X_all, prediction_all)
  invisible(gc(verbose = FALSE))
  gpu$torch$cuda$empty_cache()
}

writeLines(
  c(
    paste("completed_utc:", format(Sys.time(), tz = "UTC")),
    paste("model_tag:", model_tag),
    paste("gpu:", gpu$device_name),
    paste("training_rows:", nrow(Y_train)),
    paste("species:", ncol(Y_train)),
    paste("h5_root:", h5_root)
  ),
  file.path(log_dir, "completed.txt")
)

sjsdm_log("Completed joint sjSDM fitting and full monthly prediction.")
