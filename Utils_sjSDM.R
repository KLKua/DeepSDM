# Utilities for fitting sjSDM from the monthly species-occurrence HDF5 rasters.
#
# IMPORTANT
# ---------
# 1. Source Utils_R.R before this file. This file deliberately reuses:
#      - load_env_month()
#      - h5dataset_to_raster()
#      - check_dataset_in_h5()
# 2. No raw GBIF table is read here.
# 3. One row of the sjSDM training matrix is one sampled grid-cell x month.
#    The sampling frame is the target-group recording footprint (eta > 0),
#    reconstructed from the union of records in all available bird-species
#    occurrence rasters. If downsampling is requested, rows are sampled uniformly
#    from that footprint before the 121-species response matrix is constructed.

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

sjsdm_log <- function(...) {
  message(sprintf(
    "[%s] %s",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    paste0(..., collapse = "")
  ))
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(path)
}

assert_file_exists <- function(path, label = "file") {
  if (!file.exists(path)) {
    stop(sprintf("Required %s does not exist: %s", label, path), call. = FALSE)
  }
  invisible(path)
}

assert_sjsdm_gpu <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.", call. = FALSE)
  }
  if (!requireNamespace("sjSDM", quietly = TRUE)) {
    stop("Package 'sjSDM' is required.", call. = FALSE)
  }
  if (!sjSDM::is_torch_available()) {
    stop(
      paste0(
        "sjSDM cannot access PyTorch. Install GPU dependencies in the active ",
        "reticulate environment before running this script."
      ),
      call. = FALSE
    )
  }

  torch <- reticulate::import("torch", delay_load = FALSE)
  cuda_available <- reticulate::py_to_r(torch$cuda$is_available())
  if (!isTRUE(cuda_available)) {
    stop(
      paste0(
        "CUDA is not available to the PyTorch instance used by sjSDM. ",
        "This pipeline is GPU-only and will not fall back to CPU."
      ),
      call. = FALSE
    )
  }

  device_count <- as.integer(reticulate::py_to_r(torch$cuda$device_count()))
  device_name <- as.character(reticulate::py_to_r(torch$cuda$get_device_name(0L)))
  sjsdm_log("CUDA device 0: ", device_name, " (device_count=", device_count, ")")

  list(torch = torch, device_name = device_name, device_count = device_count)
}

balanced_month_quota <- function(total_n, dates) {
  dates <- as.character(dates)
  total_n <- as.integer(total_n)
  if (total_n < 0L) stop("total_n must be non-negative.", call. = FALSE)
  if (length(dates) == 0L) return(integer())

  quota <- rep(total_n %/% length(dates), length(dates))
  remainder <- total_n %% length(dates)
  if (remainder > 0L) quota[seq_len(remainder)] <- quota[seq_len(remainder)] + 1L
  stats::setNames(quota, dates)
}

species_h5_path <- function(species, sp_info) {
  h5_name <- sp_info$file_name[[species]]$h5file_name
  if (is.null(h5_name) || !nzchar(h5_name)) {
    stop("h5file_name not found in sp_info for species: ", species, call. = FALSE)
  }
  path <- file.path(sp_info$dir_base, h5_name)
  assert_file_exists(path, paste0("occurrence HDF5 for ", species))
  path
}

species_date_dataset <- function(species, date, sp_info) {
  dset <- sp_info$file_name[[species]]$h5file_dataset_name[[date]]
  if (is.null(dset) || !nzchar(dset)) {
    stop(
      sprintf("Occurrence dataset name missing for %s at %s", species, date),
      call. = FALSE
    )
  }
  dset
}

read_presence_cells_from_existing_h5 <- function(
    species,
    date,
    sp_info,
    extent_binary,
    allowed_cells = NULL) {

  h5_path <- species_h5_path(species, sp_info)
  dset <- species_date_dataset(species, date, sp_info)
  if (!check_dataset_in_h5(h5_path, dset)) {
    stop(
      sprintf(
        "Occurrence dataset missing for %s at %s (file=%s, dataset=%s)",
        species, date, h5_path, dset
      ),
      call. = FALSE
    )
  }

  occ_rst <- h5dataset_to_raster(h5_path, dset, template = extent_binary)
  presence_cells <- which(raster::values(occ_rst) > 0)
  if (!is.null(allowed_cells)) {
    presence_cells <- intersect(presence_cells, allowed_cells)
  }
  as.integer(presence_cells)
}

# Read presence cells for a target-group footprint species. Unlike the focal
# response reader above, this helper returns an empty vector when a species or
# month is unavailable, because some non-focal species may not have datasets for
# every month.
read_presence_cells_optional_from_existing_h5 <- function(
    species,
    date,
    sp_info,
    extent_binary,
    allowed_cells = NULL) {

  species_entry <- sp_info$file_name[[species]]
  if (is.null(species_entry)) return(integer(0))

  h5_name <- species_entry$h5file_name
  if (is.null(h5_name) || !nzchar(h5_name)) return(integer(0))

  h5_path <- file.path(sp_info$dir_base, h5_name)
  if (!file.exists(h5_path)) return(integer(0))

  dset <- species_entry$h5file_dataset_name[[date]]
  if (is.null(dset) || !nzchar(dset)) return(integer(0))
  if (!check_dataset_in_h5(h5_path, dset)) return(integer(0))

  occ_rst <- h5dataset_to_raster(h5_path, dset, template = extent_binary)
  presence_cells <- which(raster::values(occ_rst) > 0)
  if (!is.null(allowed_cells)) {
    presence_cells <- intersect(presence_cells, allowed_cells)
  }
  as.integer(presence_cells)
}

# Build one joint sjSDM training matrix from monthly occurrence rasters.
#
# Sampling design:
#   1. For each month, define the target-group recording footprint (eta > 0) as
#      the union of recorded cells across all species in footprint_species_list.
#   2. Restrict that footprint to complete-environment cells in the training
#      partition.
#   3. Use all footprint rows, or (when max_sampling_rows_total > 0) sample rows
#      uniformly from the footprint with a temporally balanced monthly quota.
#   4. Only after row selection, construct the focal N x species response matrix.
#
# Consequently, row inclusion/downsampling never depends on focal-species
# richness, rowSums(Y), or whether a row is all zero for the focal community.
build_sjsdm_training_from_occurrence_rasters <- function(
    species_list,
    date_list,
    partition_cells,
    max_sampling_rows_total = 0L,
    footprint_species_list = NULL,
    sp_info,
    extent_binary,
    env_list,
    env_info,
    DeepSDM_conf,
    seed = 42L) {

  species_list <- as.character(species_list)
  date_list <- as.character(date_list)
  partition_cells <- sort(unique(as.integer(partition_cells)))
  safe_env_names <- make.names(env_list, unique = TRUE)

  available_species <- sort(names(sp_info$file_name))
  if (is.null(footprint_species_list) || length(footprint_species_list) == 0L) {
    footprint_species_list <- available_species
  }
  footprint_species_list <- sort(unique(as.character(footprint_species_list)))

  missing_footprint_species <- setdiff(footprint_species_list, available_species)
  if (length(missing_footprint_species) > 0L) {
    warning(
      "Ignoring footprint species absent from sp_info: ",
      paste(missing_footprint_species, collapse = ", ")
    )
    footprint_species_list <- intersect(footprint_species_list, available_species)
  }
  if (length(footprint_species_list) == 0L) {
    stop("No species are available to define the target-group footprint.", call. = FALSE)
  }

  if (setequal(footprint_species_list, species_list)) {
    warning(
      paste0(
        "The target-group footprint is being defined only by the focal species. ",
        "For the intended eta > 0 design, sp_info should contain all available ",
        "bird-species occurrence rasters, not only the focal species."
      )
    )
  }

  max_sampling_rows_total <- as.integer(max_sampling_rows_total %||% 0L)
  if (max_sampling_rows_total < 0L) {
    stop("max_sampling_rows_total must be zero or positive.", call. = FALSE)
  }
  quota <- if (max_sampling_rows_total > 0L) {
    balanced_month_quota(max_sampling_rows_total, date_list)
  } else {
    stats::setNames(rep(NA_integer_, length(date_list)), date_list)
  }

  X_parts <- vector("list", length(date_list))
  Y_parts <- vector("list", length(date_list))
  metadata_parts <- vector("list", length(date_list))
  sampling_summary_parts <- vector("list", length(date_list))
  used <- 0L

  for (k in seq_along(date_list)) {
    date <- date_list[k]
    sjsdm_log("Building target-group sjSDM training rows for ", date)

    # load_env_month() applies the project's variable-specific normalization.
    env_month <- load_env_month(env_list, env_info, date, DeepSDM_conf)
    env_partition <- raster::extract(env_month, partition_cells)
    if (is.null(dim(env_partition))) {
      env_partition <- matrix(env_partition, nrow = 1L)
    }
    env_partition <- as.matrix(env_partition)
    colnames(env_partition) <- safe_env_names

    env_complete <- stats::complete.cases(env_partition)
    valid_partition_cells <- partition_cells[env_complete]
    if (length(valid_partition_cells) == 0L) {
      warning("No complete environmental cells for ", date)
      next
    }

    # Build the binary eta > 0 sampling footprint from all available bird
    # occurrence rasters. This is done before reading/constructing focal Y.
    footprint_presence_by_species <- vector("list", length(footprint_species_list))
    names(footprint_presence_by_species) <- footprint_species_list
    for (j in seq_along(footprint_species_list)) {
      footprint_presence_by_species[[j]] <-
        read_presence_cells_optional_from_existing_h5(
          species = footprint_species_list[j],
          date = date,
          sp_info = sp_info,
          extent_binary = extent_binary,
          allowed_cells = valid_partition_cells
        )
    }

    surveyed_cells <- sort(unique(unlist(
      footprint_presence_by_species,
      use.names = FALSE
    )))
    surveyed_cells <- intersect(surveyed_cells, valid_partition_cells)

    if (length(surveyed_cells) == 0L) {
      warning("No eta > 0 target-group footprint cells for ", date)
      next
    }

    n_available <- length(surveyed_cells)
    n_requested <- if (max_sampling_rows_total > 0L) {
      min(as.integer(quota[[date]]), n_available)
    } else {
      n_available
    }

    set.seed(as.integer(seed) + k)
    row_cells <- if (n_requested < n_available) {
      sort(sample(surveyed_cells, size = n_requested, replace = FALSE))
    } else {
      surveyed_cells
    }

    if (length(row_cells) == 0L) {
      warning("No sampled target-group training rows for ", date)
      next
    }

    # Extract X from the already-normalized monthly stack, preserving row order.
    X_month <- raster::extract(env_month, row_cells)
    if (is.null(dim(X_month))) X_month <- matrix(X_month, nrow = 1L)
    X_month <- as.matrix(X_month)
    colnames(X_month) <- safe_env_names

    keep <- stats::complete.cases(X_month)
    row_cells <- row_cells[keep]
    X_month <- X_month[keep, , drop = FALSE]
    if (length(row_cells) == 0L) next

    # Construct focal responses only after response-independent row selection.
    presence_by_species <- vector("list", length(species_list))
    names(presence_by_species) <- species_list
    for (j in seq_along(species_list)) {
      focal_species <- species_list[j]
      # Keep the original strict focal-response behaviour: every focal species
      # must have its expected HDF5 dataset for every training month.
      presence_by_species[[j]] <- read_presence_cells_from_existing_h5(
        species = focal_species,
        date = date,
        sp_info = sp_info,
        extent_binary = extent_binary,
        allowed_cells = valid_partition_cells
      )
    }

    Y_month <- matrix(
      0,
      nrow = length(row_cells),
      ncol = length(species_list),
      dimnames = list(NULL, species_list)
    )

    for (j in seq_along(species_list)) {
      row_index <- match(presence_by_species[[j]], row_cells)
      row_index <- row_index[!is.na(row_index)]
      if (length(row_index) > 0L) Y_month[row_index, j] <- 1
    }

    xy <- raster::xyFromCell(extent_binary, row_cells)
    sampling_fraction <- length(row_cells) / n_available
    metadata_month <- data.frame(
      row_id = paste0(row_cells, "__", gsub("-", "", date)),
      cell = row_cells,
      date = date,
      x = xy[, 1L],
      y = xy[, 2L],
      source_type = "target_group_eta_gt_zero",
      footprint_available_rows = n_available,
      sampling_fraction = sampling_fraction,
      recorded_focal_richness = rowSums(Y_month),
      stringsAsFactors = FALSE
    )

    used <- used + 1L
    X_parts[[used]] <- X_month
    Y_parts[[used]] <- Y_month
    metadata_parts[[used]] <- metadata_month
    sampling_summary_parts[[used]] <- data.frame(
      date = date,
      footprint_available_rows = n_available,
      sampled_rows = length(row_cells),
      sampling_fraction = sampling_fraction,
      sampled_rows_with_focal_record = sum(rowSums(Y_month) > 0),
      sampled_focal_all_zero_rows = sum(rowSums(Y_month) == 0),
      stringsAsFactors = FALSE
    )

    sjsdm_log(
      date, ": sampled ", length(row_cells), " of ", n_available,
      " eta > 0 target-group rows; focal non-zero rows=",
      sum(rowSums(Y_month) > 0), ", focal all-zero rows=",
      sum(rowSums(Y_month) == 0)
    )

    rm(
      env_month, env_partition, X_month, Y_month, metadata_month,
      footprint_presence_by_species, presence_by_species
    )
    invisible(gc(verbose = FALSE))
  }

  if (used == 0L) stop("No sjSDM training rows were built.", call. = FALSE)

  X <- do.call(rbind, X_parts[seq_len(used)])
  Y <- do.call(rbind, Y_parts[seq_len(used)])
  metadata <- do.call(rbind, metadata_parts[seq_len(used)])
  sampling_summary <- do.call(rbind, sampling_summary_parts[seq_len(used)])
  rownames(X) <- metadata$row_id
  rownames(Y) <- metadata$row_id

  if (nrow(X) != nrow(Y) || nrow(X) != nrow(metadata)) {
    stop("X, Y, and metadata are not aligned.", call. = FALSE)
  }
  if (!identical(colnames(Y), species_list)) {
    stop("Species order changed while building Y.", call. = FALSE)
  }
  if (!identical(colnames(X), safe_env_names)) {
    stop("Environmental predictor order changed while building X.", call. = FALSE)
  }
  if (any(!Y %in% c(0, 1))) {
    stop("Y contains values other than 0/1.", call. = FALSE)
  }

  zero_species <- colnames(Y)[colSums(Y) == 0]
  if (length(zero_species) > 0L) {
    stop(
      paste0(
        "The response-independent target-group sample contains no presence for: ",
        paste(zero_species, collapse = ", "), ". Set max_sampling_rows_total: 0 ",
        "to use all eta > 0 rows, or increase the sampling cap."
      ),
      call. = FALSE
    )
  }

  list(
    X = X,
    Y = Y,
    metadata = metadata,
    sampling_summary = sampling_summary,
    species_order = species_list,
    footprint_species_order = footprint_species_list,
    predictor_order = safe_env_names,
    original_predictor_order = env_list,
    sampling_quota = quota,
    max_sampling_rows_total = max_sampling_rows_total
  )
}

fit_joint_sjsdm_gpu <- function(
    X_train,
    Y_train,
    hidden = c(64L, 64L),
    activation = "relu",
    dropout = 0.1,
    lambda_coef = 0.01,
    alpha_coef = 0.5,
    lambda_cov = 0.01,
    alpha_cov = 0.5,
    iter = 300L,
    step_size = 4096L,
    learning_rate = 0.01,
    sampling = 100L,
    data_loader_cores = 0L,
    scheduler_patience = 20L,
    lr_reduce_factor = 0.5,
    early_stopping_training = 50L,
    seed = 42L) {

  X_train <- as.data.frame(X_train, check.names = FALSE)
  Y_train <- as.matrix(Y_train)
  step_size <- min(as.integer(step_size), nrow(Y_train))

  env_component <- sjSDM::DNN(
    data = X_train,
    formula = ~ .,
    hidden = as.integer(hidden),
    activation = activation,
    bias = TRUE,
    lambda = as.numeric(lambda_coef),
    alpha = as.numeric(alpha_coef),
    dropout = as.numeric(dropout)
  )

  control <- sjSDM::sjSDMControl(
    scheduler = as.integer(scheduler_patience),
    lr_reduce_factor = as.numeric(lr_reduce_factor),
    early_stopping_training = as.integer(early_stopping_training),
    mixed = FALSE
  )

  sjsdm_log(
    "Calling sjSDM() once for one joint model: ",
    nrow(Y_train), " rows x ", ncol(Y_train), " species"
  )

  sjSDM::sjSDM(
    Y = Y_train,
    env = env_component,
    biotic = sjSDM::bioticStruct(
      lambda = as.numeric(lambda_cov),
      alpha = as.numeric(alpha_cov),
      diag = FALSE
    ),
    family = stats::binomial("probit"),
    iter = as.integer(iter),
    step_size = step_size,
    learning_rate = as.numeric(learning_rate),
    se = FALSE,
    sampling = as.integer(sampling),
    parallel = as.integer(data_loader_cores),
    control = control,
    device = "gpu",
    dtype = "float32",
    seed = as.integer(seed),
    verbose = TRUE
  )
}

predict_joint_sjsdm_in_chunks <- function(
    model,
    X_new,
    species_order,
    chunk_size = 20000L) {

  X_new <- as.data.frame(X_new, check.names = FALSE)
  n <- nrow(X_new)
  if (n == 0L) {
    return(matrix(numeric(), nrow = 0L, ncol = length(species_order)))
  }

  starts <- seq.int(1L, n, by = as.integer(chunk_size))
  output <- matrix(
    NA_real_,
    nrow = n,
    ncol = length(species_order),
    dimnames = list(NULL, species_order)
  )

  for (start in starts) {
    end <- min(n, start + as.integer(chunk_size) - 1L)
    sjsdm_log("Predicting rows ", start, "-", end, " of ", n)
    pred <- stats::predict(
      model,
      newdata = X_new[start:end, , drop = FALSE],
      type = "link"
    )
    pred <- as.matrix(pred)
    if (ncol(pred) != length(species_order)) {
      stop(
        sprintf(
          "sjSDM returned %d species columns; expected %d.",
          ncol(pred), length(species_order)
        ),
        call. = FALSE
      )
    }
    output[start:end, ] <- pred
  }

  output
}

write_sjsdm_month_to_species_h5 <- function(
    prediction_matrix,
    species_list,
    land_cells,
    extent_binary,
    date,
    h5_root,
    overwrite = FALSE) {

  prediction_matrix <- as.matrix(prediction_matrix)
  if (nrow(prediction_matrix) != length(land_cells)) {
    stop("Prediction row count does not equal number of land cells.", call. = FALSE)
  }
  if (ncol(prediction_matrix) != length(species_list)) {
    stop("Prediction column count does not equal number of species.", call. = FALSE)
  }

  extent_vals <- raster::extent(extent_binary)
  transform_values <- c(
    extent_vals@xmin,
    raster::res(extent_binary)[1],
    0,
    extent_vals@ymax,
    0,
    -raster::res(extent_binary)[2]
  )

  for (j in seq_along(species_list)) {
    species <- species_list[j]
    species_dir <- file.path(h5_root, species)
    ensure_dir(species_dir)
    h5_path <- file.path(species_dir, sprintf("%s.h5", species))

    rst <- raster::raster(extent_binary)
    rst_values <- rep(0, raster::ncell(rst))
    rst_values[land_cells] <- prediction_matrix[, j]
    raster::values(rst) <- rst_values
    rst <- rst * extent_binary

    h5 <- hdf5r::H5File$new(h5_path, mode = "a")
    if (h5$exists(date) && !isTRUE(overwrite)) {
      h5$close()
      next
    }
    if (h5$exists(date) && isTRUE(overwrite)) {
      h5$link_delete(date)
    }

    write_error <- NULL
    tryCatch({
      h5attr(h5, "crs") <- as.character(raster::crs(extent_binary))
      h5attr(h5, "transform") <- transform_values
      h5[[date]] <- t(as.matrix(rst))
    }, error = function(e) {
      write_error <<- e
    })
    try(h5$close(), silent = TRUE)
    if (!is.null(write_error)) stop(write_error)
  }

  invisible(TRUE)
}

safe_macro_auc <- function(Y, pred) {
  Y <- as.matrix(Y)
  pred <- as.matrix(pred)
  aucs <- rep(NA_real_, ncol(Y))
  for (j in seq_len(ncol(Y))) {
    actual <- Y[, j]
    score <- pred[, j]
    keep <- is.finite(actual) & is.finite(score)
    actual <- actual[keep]
    score <- score[keep]
    if (length(unique(actual)) < 2L) next
    aucs[j] <- as.numeric(pROC::roc(actual, score, quiet = TRUE)$auc)
  }
  c(
    macro_auc = mean(aucs, na.rm = TRUE),
    n_species_with_auc = sum(is.finite(aucs))
  )
}
