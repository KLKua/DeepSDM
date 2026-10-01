#!/usr/bin/env Rscript

# Fit one joint Hmsc model and generate monthly suitability rasters.
#
# Usage:
#   Rscript run_hmsc.R hmsc_conf.yml
#
# This is the Hmsc replacement for run_sjsdm_gpu.R. It keeps the same project
# inputs and HDF5 output interface, but uses a simple presence-background design:
#   - uniform terrestrial cell-month backgrounds, balanced across months;
#   - at background rows: all focal species are 0, except recorded species = 1;
#   - at presence-selected rows: recorded species = 1, other species = NA;
#   - no target-group footprint, effort offset, traits, or phylogeny;
#   - one joint probit Hmsc with a sampling-unit latent random level;
#   - unconditional full-grid predictions, without test-community occurrences.

suppressPackageStartupMessages({
  library(Hmsc)
  library(coda)
  library(raster)
  library(hdf5r)
  library(yaml)
  library(rjson)
})

source("Utils_R.R")
source("Utils_Hmsc.R")
assert_packages(c("Hmsc", "coda", "raster", "hdf5r", "yaml", "rjson"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  stop("Usage: Rscript run_hmsc.R hmsc_conf.yml", call. = FALSE)
}

config_path <- args[1L]
assert_file_exists(config_path, "Hmsc YAML configuration")
cfg <- yaml::read_yaml(config_path)

run_id <- assert_nonempty_character(cfg$run_id, "run_id")
exp_id <- assert_nonempty_character(cfg$exp_id, "exp_id")
model_tag <- assert_nonempty_character(
  cfg$model_tag %||% "hmsc_probit_uniform_background",
  "model_tag"
)

if (utils::compareVersion(
  as.character(utils::packageVersion("Hmsc")), "3.3.7"
) < 0L) {
  stop("Hmsc >= 3.3-7 is required.", call. = FALSE)
}

# -----------------------------------------------------------------------------
# 1. Load the existing DeepSDM project configuration and spatial partition
# -----------------------------------------------------------------------------
DeepSDM_conf_path <- file.path("predicts", run_id, "DeepSDM_conf.yaml")
assert_file_exists(DeepSDM_conf_path, "DeepSDM configuration")
DeepSDM_conf <- yaml::read_yaml(DeepSDM_conf_path)

env_list <- sort(assert_nonempty_character(unlist(
  DeepSDM_conf$training_conf$env_list, use.names = FALSE
), "env_list"))
date_list_train <- assert_nonempty_character(unlist(
  DeepSDM_conf$training_conf$date_list_train, use.names = FALSE
), "date_list_train")
date_list_predict <- assert_nonempty_character(unlist(
  DeepSDM_conf$training_conf$date_list_predict, use.names = FALSE
), "date_list_predict")
species_list_train <- sort(assert_nonempty_character(unlist(
  DeepSDM_conf$training_conf$species_list_train, use.names = FALSE
), "species_list_train"))
species_list_predict <- sort(assert_nonempty_character(unlist(
  DeepSDM_conf$training_conf$species_list_predict, use.names = FALSE
), "species_list_predict"))

if (!identical(species_list_train, species_list_predict)) {
  stop(
    "species_list_train and species_list_predict differ; one joint Hmsc requires ",
    "a fixed species column order.",
    call. = FALSE
  )
}
species_list <- species_list_train

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

hmsc_log(
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
  cfg$output_root %||% file.path("predicts_hmsc", run_id)
)
model_root <- ensure_dir(file.path(output_root, model_tag))
data_dir <- ensure_dir(file.path(model_root, "data"))
model_dir <- ensure_dir(file.path(model_root, "model"))
h5_root <- ensure_dir(file.path(model_root, "h5", "all"))
log_dir <- ensure_dir(file.path(model_root, "logs"))

training_rds <- file.path(data_dir, "joint_training_uniform_background.rds")
model_path <- file.path(model_dir, "hmsc_model.rds")
rebuild_training_data <- isTRUE(cfg$rebuild_training_data %||% FALSE)
refit_model <- isTRUE(cfg$refit_model %||% FALSE)
overwrite_predictions <- isTRUE(cfg$prediction$overwrite %||% FALSE)

# -----------------------------------------------------------------------------
# 3. Existing occurrence rasters -> one partially observed joint community matrix
# -----------------------------------------------------------------------------
if (rebuild_training_data || !file.exists(training_rds)) {
  training_data <- build_hmsc_uniform_background_training(
    species_list = species_list,
    date_list = date_list_train,
    partition_cells = i_trainsplit,
    n_background_rows_total = assert_positive_integer(
      cfg$training_data$n_background_rows_total %||% 10000L,
      "training_data.n_background_rows_total"
    ),
    max_presence_rows_total = assert_positive_integer(
      cfg$training_data$max_presence_rows_total %||% 10000L,
      "training_data.max_presence_rows_total",
      allow_zero = TRUE
    ),
    sp_info = sp_info,
    extent_binary = extent_binary,
    env_list = env_list,
    env_info = env_info,
    DeepSDM_conf = DeepSDM_conf,
    seed = as.integer(cfg$training_data$seed %||% 42L)
  )
  saveRDS(training_data, training_rds, compress = FALSE)
} else {
  hmsc_log("Loading cached Hmsc training matrix: ", training_rds)
  training_data <- readRDS(training_rds)
}

X_train <- as.data.frame(training_data$X, check.names = FALSE)
Y_train <- as.matrix(training_data$Y)
row_ids <- as.character(training_data$metadata$row_id)

if (!identical(colnames(Y_train), species_list)) {
  stop("Cached response matrix has a different species order.", call. = FALSE)
}
if (!identical(colnames(X_train), training_data$predictor_order)) {
  stop("Cached predictor order is inconsistent.", call. = FALSE)
}
if (nrow(X_train) != nrow(Y_train) || nrow(Y_train) != length(row_ids)) {
  stop("X_train, Y_train, and metadata have different row counts.", call. = FALSE)
}
if (!identical(rownames(X_train), row_ids) || !identical(rownames(Y_train), row_ids)) {
  stop("X_train, Y_train, and metadata row identifiers are not aligned.",
       call. = FALSE)
}
if (anyDuplicated(row_ids)) {
  stop("The Hmsc training matrix contains duplicated cell-month rows.",
       call. = FALSE)
}
if (any(training_data$metadata$cell %in% i_valsplit)) {
  stop("Validation cells leaked into the Hmsc training matrix.", call. = FALSE)
}

write.csv(
  training_data$metadata,
  file.path(data_dir, "joint_training_row_metadata.csv"),
  row.names = FALSE
)
write.csv(
  training_data$background_summary,
  file.path(data_dir, "uniform_background_sampling_summary.csv"),
  row.names = FALSE
)
write.csv(
  training_data$species_summary,
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
    )),
    stringsAsFactors = FALSE
  ),
  file.path(data_dir, "environment_preprocessing_audit.csv"),
  row.names = FALSE
)
yaml::write_yaml(
  list(
    design = training_data$design,
    target_group_background_used = FALSE,
    n_rows = nrow(Y_train),
    n_species = ncol(Y_train),
    n_predictors = ncol(X_train),
    n_uniform_background_rows = sum(
      training_data$metadata$source_type == "uniform_background"
    ),
    n_presence_selected_rows = sum(
      training_data$metadata$source_type == "presence_only"
    ),
    n_zero_labels = sum(Y_train == 0, na.rm = TRUE),
    n_one_labels = sum(Y_train == 1, na.rm = TRUE),
    n_missing_labels = sum(is.na(Y_train))
  ),
  file.path(data_dir, "training_design_audit.yml")
)

hmsc_log(
  "Training matrix: ", nrow(Y_train), " cell-month rows x ",
  ncol(Y_train), " species; predictors=", ncol(X_train),
  "; observed 0/1/NA=", sum(Y_train == 0, na.rm = TRUE), "/",
  sum(Y_train == 1, na.rm = TRUE), "/", sum(is.na(Y_train))
)

# -----------------------------------------------------------------------------
# 4. Fit or load exactly one joint probit Hmsc
# -----------------------------------------------------------------------------
if (file.exists(model_path) && !refit_model) {
  if (file.mtime(model_path) < file.mtime(training_rds)) {
    stop(
      "The cached Hmsc model is older than the training matrix. Set ",
      "refit_model: true.",
      call. = FALSE
    )
  }
  hmsc_log("Loading cached Hmsc model: ", model_path)
  model <- readRDS(model_path)
} else {
  if (file.exists(model_path) && refit_model && !overwrite_predictions) {
    existing_h5 <- list.files(
      h5_root, pattern = "\\.h5$", recursive = TRUE, full.names = TRUE
    )
    if (length(existing_h5) > 0L) {
      stop(
        "Refitting would make existing Hmsc predictions stale. Set ",
        "prediction.overwrite: true or remove the old HDF5 files.",
        call. = FALSE
      )
    }
  }

  latent_factors <- assert_positive_integer(
    cfg$model$latent_factors %||% 2L,
    "model.latent_factors"
  )
  studyDesign <- data.frame(
    sample = factor(row_ids, levels = row_ids),
    stringsAsFactors = TRUE,
    row.names = row_ids
  )
  rL_sample <- Hmsc::HmscRandomLevel(units = levels(studyDesign$sample))
  rL_sample <- Hmsc::setPriors(
    rL_sample,
    nfMin = latent_factors,
    nfMax = latent_factors
  )

  model <- Hmsc::Hmsc(
    Y = Y_train,
    XData = X_train,
    XFormula = ~ .,
    XScale = FALSE,
    distr = "probit",
    studyDesign = studyDesign,
    ranLevels = list(sample = rL_sample)
  )

  samples <- assert_positive_integer(cfg$model$samples %||% 500L, "model.samples")
  transient <- assert_positive_integer(
    cfg$model$transient %||% 1000L,
    "model.transient",
    allow_zero = TRUE
  )
  thin <- assert_positive_integer(cfg$model$thin %||% 5L, "model.thin")
  n_chains <- assert_positive_integer(cfg$model$n_chains %||% 4L, "model.n_chains")
  n_parallel <- min(
    n_chains,
    assert_positive_integer(cfg$model$n_parallel %||% n_chains, "model.n_parallel")
  )
  use_socket <- isTRUE(cfg$model$use_socket %||% FALSE)
  verbose <- as.integer(cfg$model$verbose %||% 100L)
  model_seed <- as.integer(cfg$model$seed %||% 20260302L)
  set.seed(model_seed)

  hmsc_log(
    "Fitting one Hmsc model: rows=", nrow(Y_train),
    ", species=", ncol(Y_train),
    ", predictors=", ncol(X_train),
    ", latent_factors=", latent_factors,
    ", chains=", n_chains
  )
  fit_start <- Sys.time()
  model <- Hmsc::sampleMcmc(
    model,
    samples = samples,
    transient = transient,
    thin = thin,
    nChains = n_chains,
    nParallel = n_parallel,
    useSocket = use_socket,
    alignPost = TRUE,
    verbose = verbose
  )
  fit_elapsed <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))
  saveRDS(model, model_path, compress = FALSE)

  beta_post <- Hmsc::getPostEstimate(model, parName = "Beta")
  beta_names <- rownames(beta_post$mean)
  if (is.null(beta_names) || length(beta_names) != nrow(beta_post$mean) ||
      any(!nzchar(beta_names))) {
    beta_names <- model$covNames
  }
  rownames(beta_post$mean) <- beta_names
  write.csv(
    data.frame(predictor = beta_names, beta_post$mean, check.names = FALSE),
    file.path(model_dir, "posterior_mean_beta.csv"),
    row.names = FALSE
  )

  hmsc_log("Calculating MCMC diagnostics for Beta and Lambda.")
  diagnostics <- hmsc_convergence_table(model, include_lambda = TRUE)
  write.csv(
    diagnostics,
    file.path(model_dir, "mcmc_diagnostics_beta_lambda.csv"),
    row.names = FALSE
  )

  yaml::write_yaml(
    list(
      run_id = run_id,
      exp_id = exp_id,
      model_tag = model_tag,
      design = training_data$design,
      target_group_background_used = FALSE,
      distribution = "probit",
      n_training_rows = nrow(Y_train),
      n_species = ncol(Y_train),
      n_predictors = ncol(X_train),
      latent_factors = latent_factors,
      samples = samples,
      transient = transient,
      thin = thin,
      n_chains = n_chains,
      n_parallel = n_parallel,
      fit_elapsed_seconds = fit_elapsed,
      fraction_rhat_le_1_1 = safe_fraction_true(diagnostics$rhat <= 1.1),
      minimum_effective_sample_size = finite_min(
        diagnostics$effective_sample_size
      ),
      species_order = species_list,
      predictor_order = colnames(X_train),
      hmsc_version = as.character(utils::packageVersion("Hmsc"))
    ),
    file.path(model_dir, "fit_record.yml")
  )
  capture.output(sessionInfo(), file = file.path(log_dir, "sessionInfo_fit.txt"))
}

prediction_spec <- prepare_hmsc_prediction(model)
if (!identical(prediction_spec$species_order, species_list)) {
  stop("The fitted Hmsc species order differs from the project species order.",
       call. = FALSE)
}

smoke_n <- min(20L, nrow(X_train))
smoke <- predict_hmsc_fixed_mean(
  prediction_spec,
  X_train[seq_len(smoke_n), , drop = FALSE],
  chunk_size = smoke_n
)
if (any(!is.finite(smoke))) {
  stop("The fitted Hmsc failed the prediction smoke test.", call. = FALSE)
}

# -----------------------------------------------------------------------------
# 5. Apply the fitted model to all terrestrial cells for every month
# -----------------------------------------------------------------------------
prediction_chunk_size <- assert_positive_integer(
  cfg$prediction$chunk_size %||% 20000L,
  "prediction.chunk_size"
)
safe_env_names <- training_data$predictor_order

for (date in date_list_predict) {
  if (!overwrite_predictions) {
    month_complete <- all(vapply(species_list, function(sp) {
      path <- file.path(h5_root, sp, sprintf("%s.h5", sp))
      file.exists(path) && check_dataset_in_h5(path, date)
    }, logical(1)))
    if (month_complete) {
      hmsc_log("Skipping completed month: ", date)
      next
    }
  }

  hmsc_log("Predicting all terrestrial cells for ", date)
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
    prediction_all[valid, ] <- predict_hmsc_fixed_mean(
      prediction_spec,
      X_all[valid, , drop = FALSE],
      chunk_size = prediction_chunk_size
    )
  }

  write_hmsc_month_to_species_h5(
    prediction_matrix = prediction_all,
    species_list = species_list,
    land_cells = i_extent,
    extent_binary = extent_binary,
    date = date,
    h5_root = h5_root,
    overwrite = overwrite_predictions
  )

  rm(env_month, X_all, prediction_all)
  invisible(gc(verbose = FALSE))
}

writeLines(
  c(
    paste("completed_utc:", format(Sys.time(), tz = "UTC")),
    paste("model_tag:", model_tag),
    paste("training_rows:", nrow(Y_train)),
    paste("species:", ncol(Y_train)),
    paste("target_group_background_used:", FALSE),
    paste("h5_root:", h5_root)
  ),
  file.path(log_dir, "completed.txt")
)
capture.output(sessionInfo(), file = file.path(log_dir, "sessionInfo_prediction.txt"))
hmsc_log("Completed joint Hmsc fitting and full monthly prediction.")
