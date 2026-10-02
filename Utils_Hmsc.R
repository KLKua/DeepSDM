# Utilities for the Hmsc JSDM replacement template.
# Source Utils_R.R before this file.
#
# Design:
#   * one row = one raster-cell x month sampling unit;
#   * uniformly sampled terrestrial background rows: 0 for every species,
#     overwritten to 1 where a focal occurrence is present;
#   * presence-selected rows: recorded focal species = 1 and all other focal
#     species = NA, so one species' record is not a target-group negative for
#     another species;
#   * no effort offset, traits, phylogeny, or target-group footprint.

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

hmsc_log <- function(...) {
  message(sprintf(
    "[%s] %s",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    paste0(..., collapse = "")
  ))
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

assert_file_exists <- function(path, label = "file") {
  if (!file.exists(path)) {
    stop(sprintf("Required %s does not exist: %s", label, path), call. = FALSE)
  }
  invisible(path)
}

assert_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0L) {
    stop(
      "Missing R package(s): ", paste(missing, collapse = ", "),
      ". Install them before running the Hmsc pipeline.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

assert_positive_integer <- function(x, label, allow_zero = FALSE) {
  value <- suppressWarnings(as.integer(x))
  lower <- if (isTRUE(allow_zero)) 0L else 1L
  if (length(value) != 1L || is.na(value) || value < lower) {
    stop(
      sprintf("%s must be one integer >= %d.", label, lower),
      call. = FALSE
    )
  }
  value
}

assert_nonempty_character <- function(x, label) {
  x <- as.character(x)
  if (length(x) == 0L || anyNA(x) || any(!nzchar(x))) {
    stop(label, " must contain non-empty character values.", call. = FALSE)
  }
  x
}

assert_raster_alignment <- function(reference, candidate, candidate_label = "raster") {
  aligned <- raster::compareRaster(
    reference, candidate,
    extent = TRUE, rowcol = TRUE, crs = TRUE, res = TRUE,
    orig = TRUE, rotation = TRUE,
    stopiffalse = FALSE
  )
  if (!isTRUE(aligned)) {
    stop(candidate_label, " is not aligned with the project extent raster.", call. = FALSE)
  }
  invisible(TRUE)
}

balanced_quota <- function(total_n, groups) {
  groups <- as.character(groups)
  total_n <- as.integer(total_n)
  if (total_n < 0L) stop("total_n must be non-negative.", call. = FALSE)
  if (length(groups) == 0L) return(integer())
  out <- rep(total_n %/% length(groups), length(groups))
  rem <- total_n %% length(groups)
  if (rem > 0L) out[seq_len(rem)] <- out[seq_len(rem)] + 1L
  stats::setNames(out, groups)
}

stable_seed <- function(base_seed, ...) {
  txt <- paste(c(base_seed, ...), collapse = "::")
  ints <- utf8ToInt(txt)
  value <- (as.double(base_seed) + sum(as.double(ints) * seq_along(ints))) %% 2147483000
  as.integer(value + 1L)
}

finite_min <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else min(x)
}

safe_fraction_true <- function(x) {
  x <- as.logical(x)
  x <- x[!is.na(x)]
  if (length(x) == 0L) NA_real_ else mean(x)
}

make_hmsc_row_key <- function(cell, date) paste0(date, "__", as.integer(cell))
row_key_date <- function(keys) sub("__[0-9]+$", "", as.character(keys))
row_key_cell <- function(keys) as.integer(sub("^.*__", "", as.character(keys)))

species_h5_path_hmsc <- function(species, sp_info) {
  entry <- sp_info$file_name[[species]]
  if (is.null(entry)) stop("Species missing from sp_info: ", species, call. = FALSE)
  h5_name <- entry$h5file_name
  if (is.null(h5_name) || !nzchar(h5_name)) {
    stop("h5file_name missing in sp_info for: ", species, call. = FALSE)
  }
  path <- file.path(sp_info$dir_base, h5_name)
  assert_file_exists(path, paste0("occurrence HDF5 for ", species))
  path
}

species_date_dataset_hmsc <- function(species, date, sp_info) {
  entry <- sp_info$file_name[[species]]
  dset <- if (is.null(entry)) NULL else entry$h5file_dataset_name[[date]]
  if (is.null(dset) || !nzchar(dset)) {
    stop(sprintf("Occurrence dataset name missing for %s at %s", species, date),
         call. = FALSE)
  }
  dset
}

read_presence_cells_hmsc <- function(
    species, date, sp_info, extent_binary, allowed_cells = NULL) {
  h5_path <- species_h5_path_hmsc(species, sp_info)
  dset <- species_date_dataset_hmsc(species, date, sp_info)
  if (!check_dataset_in_h5(h5_path, dset)) {
    stop(
      sprintf("Occurrence dataset missing for %s at %s: %s/%s",
              species, date, h5_path, dset),
      call. = FALSE
    )
  }
  rst <- h5dataset_to_raster(h5_path, dset, template = extent_binary)
  cells <- which(raster::values(rst) > 0)
  if (!is.null(allowed_cells)) cells <- intersect(cells, allowed_cells)
  as.integer(cells)
}

sample_presence_keys_balanced <- function(
    keys_by_species, max_total = 0L, seed = 42L) {
  species <- names(keys_by_species)
  keys_by_species <- lapply(keys_by_species, function(x) unique(as.character(x)))
  available <- unique(unlist(keys_by_species, use.names = FALSE))
  max_total <- as.integer(max_total)

  if (length(available) == 0L) stop("No focal presence rows are available.", call. = FALSE)
  if (max_total <= 0L || length(available) <= max_total) return(available)

  species_with_data <- species[lengths(keys_by_species) > 0L]
  if (max_total < length(species_with_data)) {
    stop(
      "max_presence_rows_total must be at least the number of species with data (",
      length(species_with_data), ").",
      call. = FALSE
    )
  }

  quota <- balanced_quota(max_total, species_with_data)
  selected <- character()
  for (sp in species_with_data) {
    candidates <- keys_by_species[[sp]]
    n_take <- min(length(candidates), quota[[sp]])
    set.seed(stable_seed(seed, "presence", sp))
    selected <- c(selected, sample(candidates, n_take, replace = FALSE))
  }
  selected <- unique(selected)

  if (length(selected) < max_total) {
    remaining <- setdiff(available, selected)
    n_extra <- min(max_total - length(selected), length(remaining))
    if (n_extra > 0L) {
      set.seed(stable_seed(seed, "presence-extra"))
      selected <- c(selected, sample(remaining, n_extra, replace = FALSE))
    }
  }
  unique(selected)
}

# Build the partially observed Hmsc response matrix from existing monthly HDF5
# occurrence rasters. No raw GBIF table is read.
build_hmsc_uniform_background_training <- function(
    species_list,
    date_list,
    partition_cells,
    n_background_rows_total,
    max_presence_rows_total,
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
  n_background_rows_total <- as.integer(n_background_rows_total)
  max_presence_rows_total <- as.integer(max_presence_rows_total)

  if (n_background_rows_total <= 0L) {
    stop("n_background_rows_total must be positive.", call. = FALSE)
  }
  missing_species <- setdiff(species_list, names(sp_info$file_name))
  if (length(missing_species) > 0L) {
    stop("Focal species absent from sp_info: ", paste(missing_species, collapse = ", "),
         call. = FALSE)
  }

  valid_cells_by_date <- stats::setNames(vector("list", length(date_list)), date_list)
  presence_cells_by_date <- stats::setNames(vector("list", length(date_list)), date_list)
  keys_by_species <- stats::setNames(
    lapply(species_list, function(x) character()),
    species_list
  )

  # Pass 1a: identify complete-environment training cells for every month.
  for (date in date_list) {
    hmsc_log("Indexing complete-environment training cells for ", date)
    env_month <- load_env_month(env_list, env_info, date, DeepSDM_conf)
    env_partition <- raster::extract(env_month, partition_cells)
    if (is.null(dim(env_partition))) env_partition <- matrix(env_partition, nrow = 1L)
    valid_cells_by_date[[date]] <- partition_cells[stats::complete.cases(env_partition)]
    presence_cells_by_date[[date]] <- stats::setNames(
      vector("list", length(species_list)), species_list
    )
    rm(env_month, env_partition)
    invisible(gc(verbose = FALSE))
  }

  # Pass 1b: open each species HDF5 once and read its complete monthly history.
  # This avoids tens of thousands of repeated HDF5 open/close operations.
  expected_dim <- c(raster::nrow(extent_binary), raster::ncol(extent_binary))
  for (sp in species_list) {
    hmsc_log("Indexing focal occurrences for ", sp)
    h5_path <- species_h5_path_hmsc(sp, sp_info)
    h5 <- hdf5r::H5File$new(h5_path, mode = "r")
    read_error <- NULL
    tryCatch({
      for (date in date_list) {
        dset <- species_date_dataset_hmsc(sp, date, sp_info)
        if (!h5$exists(dset)) {
          stop(sprintf(
            "Occurrence dataset missing for %s at %s: %s/%s",
            sp, date, h5_path, dset
          ))
        }
        arr <- h5[[dset]]
        mat <- t(arr[seq_len(arr$dims[1L]), seq_len(arr$dims[2L])])
        try(arr$close(), silent = TRUE)
        if (!identical(as.integer(dim(mat)), as.integer(expected_dim))) {
          stop(sprintf(
            "Occurrence raster dimensions differ from extent for %s at %s.",
            sp, date
          ))
        }
        rst <- raster::raster(mat)
        cells <- which(raster::values(rst) > 0)
        cells <- intersect(cells, valid_cells_by_date[[date]])
        cells <- as.integer(cells)
        presence_cells_by_date[[date]][[sp]] <- cells
        if (length(cells) > 0L) {
          keys_by_species[[sp]] <- c(
            keys_by_species[[sp]], make_hmsc_row_key(cells, date)
          )
        }
      }
    }, error = function(e) read_error <<- e)
    try(h5$close(), silent = TRUE)
    if (!is.null(read_error)) stop(read_error)
    invisible(gc(verbose = FALSE))
  }

  zero_species <- names(keys_by_species)[lengths(keys_by_species) == 0L]
  if (length(zero_species) > 0L) {
    stop(
      "No training-partition presence was found for: ",
      paste(zero_species, collapse = ", "),
      call. = FALSE
    )
  }

  selected_presence_keys <- sample_presence_keys_balanced(
    keys_by_species = keys_by_species,
    max_total = max_presence_rows_total,
    seed = seed
  )

  # Uniform terrestrial background, temporally balanced and independent of Y.
  background_quota <- balanced_quota(n_background_rows_total, date_list)
  background_keys <- character()
  background_summary <- vector("list", length(date_list))
  for (k in seq_along(date_list)) {
    date <- date_list[k]
    candidates <- valid_cells_by_date[[date]]
    requested <- as.integer(background_quota[[date]])
    n_take <- min(requested, length(candidates))
    if (n_take < requested) {
      warning(sprintf("%s: requested %d backgrounds but only %d are available.",
                      date, requested, length(candidates)))
    }
    set.seed(stable_seed(seed, "background", date))
    cells <- if (n_take <= 0L) {
      integer()
    } else if (n_take == length(candidates)) {
      candidates
    } else {
      candidates[sample.int(length(candidates), n_take, replace = FALSE)]
    }
    background_keys <- c(background_keys, make_hmsc_row_key(cells, date))
    background_summary[[k]] <- data.frame(
      date = date,
      complete_training_cells = length(candidates),
      requested_background_rows = requested,
      sampled_background_rows = n_take,
      stringsAsFactors = FALSE
    )
  }
  background_keys <- unique(background_keys)

  selected_presence_by_date <- split(
    selected_presence_keys,
    factor(row_key_date(selected_presence_keys), levels = date_list)
  )
  background_by_date <- split(
    background_keys,
    factor(row_key_date(background_keys), levels = date_list)
  )

  X_parts <- list()
  Y_parts <- list()
  metadata_parts <- list()
  used <- 0L

  # Pass 2: build X and partially observed Y for selected rows only.
  for (date in date_list) {
    presence_keys_date <- selected_presence_by_date[[date]] %||% character()
    background_keys_date <- background_by_date[[date]] %||% character()
    row_cells <- sort(unique(c(
      row_key_cell(presence_keys_date),
      row_key_cell(background_keys_date)
    )))
    row_cells <- row_cells[is.finite(row_cells)]
    if (length(row_cells) == 0L) next

    row_keys <- make_hmsc_row_key(row_cells, date)
    is_background <- row_keys %in% background_keys_date

    env_month <- load_env_month(env_list, env_info, date, DeepSDM_conf)
    X_month <- raster::extract(env_month, row_cells)
    if (is.null(dim(X_month))) X_month <- matrix(X_month, nrow = 1L)
    X_month <- as.matrix(X_month)
    colnames(X_month) <- safe_env_names
    keep <- stats::complete.cases(X_month)
    row_cells <- row_cells[keep]
    row_keys <- row_keys[keep]
    is_background <- is_background[keep]
    X_month <- X_month[keep, , drop = FALSE]
    if (length(row_cells) == 0L) next

    Y_month <- matrix(
      NA_real_, nrow = length(row_cells), ncol = length(species_list),
      dimnames = list(row_keys, species_list)
    )
    Y_month[is_background, ] <- 0

    for (j in seq_along(species_list)) {
      sp <- species_list[j]
      idx <- match(presence_cells_by_date[[date]][[sp]], row_cells)
      idx <- idx[!is.na(idx)]
      if (length(idx) > 0L) Y_month[idx, j] <- 1
    }

    xy <- raster::xyFromCell(extent_binary, row_cells)
    metadata_month <- data.frame(
      row_id = row_keys,
      cell = row_cells,
      date = date,
      x = xy[, 1L],
      y = xy[, 2L],
      source_type = ifelse(is_background, "uniform_background", "presence_only"),
      recorded_focal_richness = rowSums(Y_month == 1, na.rm = TRUE),
      n_observed_responses = rowSums(!is.na(Y_month)),
      stringsAsFactors = FALSE
    )

    used <- used + 1L
    X_parts[[used]] <- X_month
    Y_parts[[used]] <- Y_month
    metadata_parts[[used]] <- metadata_month
    rm(env_month, X_month, Y_month, metadata_month)
    invisible(gc(verbose = FALSE))
  }

  if (used == 0L) stop("No Hmsc training rows were built.", call. = FALSE)
  X <- do.call(rbind, X_parts)
  Y <- do.call(rbind, Y_parts)
  metadata <- do.call(rbind, metadata_parts)
  rownames(X) <- metadata$row_id
  rownames(Y) <- metadata$row_id

  if (nrow(X) != nrow(Y) || nrow(Y) != nrow(metadata)) {
    stop("X, Y, and metadata are not aligned.", call. = FALSE)
  }
  if (!identical(colnames(Y), species_list)) stop("Species order changed.", call. = FALSE)
  if (!identical(colnames(X), safe_env_names)) stop("Predictor order changed.", call. = FALSE)
  observed_values <- Y[!is.na(Y)]
  if (any(!observed_values %in% c(0, 1))) stop("Y contains values outside 0/1/NA.")
  if (any(rowSums(!is.na(Y)) == 0L)) stop("A training row has no observed response.")
  if (any(colSums(Y == 1, na.rm = TRUE) == 0L)) stop("A species has no presence label.")
  if (any(colSums(Y == 0, na.rm = TRUE) == 0L)) stop("A species has no background label.")

  species_summary <- data.frame(
    species = species_list,
    available_presence_rows = vapply(keys_by_species, function(x) length(unique(x)), integer(1)),
    selected_presence_rows = vapply(
      keys_by_species,
      function(x) sum(unique(x) %in% selected_presence_keys),
      integer(1)
    ),
    model_presence_labels = colSums(Y == 1, na.rm = TRUE),
    model_background_labels = colSums(Y == 0, na.rm = TRUE),
    model_missing_labels = colSums(is.na(Y)),
    stringsAsFactors = FALSE
  )

  list(
    X = X,
    Y = Y,
    metadata = metadata,
    species_summary = species_summary,
    background_summary = do.call(rbind, background_summary),
    species_order = species_list,
    predictor_order = safe_env_names,
    original_predictor_order = env_list,
    selected_presence_keys = selected_presence_keys,
    background_keys = background_keys,
    design = "uniform_background_plus_partial_presence_rows",
    seed = as.integer(seed)
  )
}

write_hmsc_month_to_species_h5 <- function(
    prediction_matrix, species_list, land_cells, extent_binary, date,
    h5_root, overwrite = FALSE) {
  prediction_matrix <- as.matrix(prediction_matrix)
  if (nrow(prediction_matrix) != length(land_cells)) {
    stop("Prediction row count differs from number of terrestrial cells.")
  }
  if (ncol(prediction_matrix) != length(species_list)) {
    stop("Prediction column count differs from number of species.")
  }

  ex <- raster::extent(extent_binary)
  transform_values <- c(
    ex@xmin, raster::res(extent_binary)[1], 0,
    ex@ymax, 0, -raster::res(extent_binary)[2]
  )

  for (j in seq_along(species_list)) {
    species <- species_list[j]
    species_dir <- ensure_dir(file.path(h5_root, species))
    h5_path <- file.path(species_dir, sprintf("%s.h5", species))
    h5 <- hdf5r::H5File$new(h5_path, mode = "a")
    if (h5$exists(date) && !isTRUE(overwrite)) {
      h5$close()
      next
    }
    if (h5$exists(date)) h5$link_delete(date)

    rst <- raster::raster(extent_binary)
    vals <- rep(0, raster::ncell(rst))
    vals[land_cells] <- prediction_matrix[, j]
    raster::values(rst) <- vals
    rst <- rst * extent_binary

    err <- NULL
    tryCatch({
      hdf5r::h5attr(h5, "crs") <- as.character(raster::crs(extent_binary))
      hdf5r::h5attr(h5, "transform") <- transform_values
      h5[[date]] <- t(as.matrix(rst))
    }, error = function(e) err <<- e)
    try(h5$close(), silent = TRUE)
    if (!is.null(err)) stop(err)
  }
  invisible(TRUE)
}

# Prepare the posterior-mean environmental component once, then reuse it for
# every month. New sampling-unit latent effects are set to their unconditional
# mean (zero), so test-region occurrences of other species are never supplied.
prepare_hmsc_prediction <- function(model) {
  beta <- Hmsc::getPostEstimate(model, parName = "Beta")$mean
  beta_names <- rownames(beta)
  if (is.null(beta_names) || length(beta_names) != nrow(beta) || any(!nzchar(beta_names))) {
    beta_names <- model$covNames
  }
  if (is.null(beta_names) || length(beta_names) != nrow(beta)) {
    stop("Cannot determine Hmsc coefficient names.", call. = FALSE)
  }
  rownames(beta) <- beta_names
  species_order <- colnames(beta) %||% model$spNames
  if (is.null(species_order) || length(species_order) != ncol(beta)) {
    stop("Cannot determine Hmsc species names.", call. = FALSE)
  }
  colnames(beta) <- species_order
  list(
    beta = beta,
    beta_names = beta_names,
    species_order = species_order,
    XFormula = model$XFormula
  )
}

predict_hmsc_fixed_mean <- function(
    prediction_spec, XData, chunk_size = 20000L) {
  XData <- as.data.frame(XData, check.names = FALSE)
  n <- nrow(XData)
  if (n == 0L) {
    return(matrix(
      numeric(), nrow = 0L,
      ncol = length(prediction_spec$species_order),
      dimnames = list(NULL, prediction_spec$species_order)
    ))
  }

  design <- stats::model.matrix(prediction_spec$XFormula, data = XData)
  missing_cols <- setdiff(prediction_spec$beta_names, colnames(design))
  if (length(missing_cols) > 0L) {
    stop(
      "Prediction design is missing: ", paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  design <- design[, prediction_spec$beta_names, drop = FALSE]

  chunk_size <- max(1L, as.integer(chunk_size))
  output <- matrix(
    NA_real_, nrow = n, ncol = length(prediction_spec$species_order),
    dimnames = list(NULL, prediction_spec$species_order)
  )
  for (start in seq.int(1L, n, by = chunk_size)) {
    end <- min(n, start + chunk_size - 1L)
    hmsc_log("Predicting rows ", start, "-", end, " of ", n)
    output[start:end, ] <- stats::pnorm(
      design[start:end, , drop = FALSE] %*% prediction_spec$beta
    )
  }
  output
}

matrix_parameter_names <- function(x, prefix) {
  rn <- rownames(x) %||% as.character(seq_len(nrow(x)))
  cn <- colnames(x) %||% as.character(seq_len(ncol(x)))
  as.vector(outer(rn, cn, function(r, c) paste(prefix, r, c, sep = "::")))
}

extract_hmsc_parameter_chains <- function(model, include_lambda = TRUE) {
  chains <- lapply(model$postList, function(chain) {
    rows <- lapply(chain, function(sam) {
      beta <- sam$Beta
      values <- stats::setNames(as.numeric(beta), matrix_parameter_names(beta, "Beta"))
      if (isTRUE(include_lambda) && length(sam$Lambda) > 0L) {
        lambda <- sam$Lambda[[1L]]
        values <- c(
          values,
          stats::setNames(as.numeric(lambda), matrix_parameter_names(lambda, "Lambda"))
        )
      }
      values
    })
    coda::mcmc(do.call(rbind, rows))
  })
  coda::mcmc.list(chains)
}

hmsc_convergence_table <- function(model, include_lambda = TRUE) {
  mlist <- extract_hmsc_parameter_chains(model, include_lambda = include_lambda)
  ess <- tryCatch(coda::effectiveSize(mlist), error = function(e) numeric())
  rhat <- if (length(mlist) >= 2L) {
    tryCatch(
      coda::gelman.diag(
        mlist, autoburnin = FALSE, multivariate = FALSE
      )$psrf[, "Point est."],
      error = function(e) numeric()
    )
  } else {
    stats::setNames(rep(NA_real_, length(ess)), names(ess))
  }
  pars <- union(names(ess), names(rhat))
  data.frame(
    parameter = pars,
    effective_sample_size = as.numeric(ess[pars]),
    rhat = as.numeric(rhat[pars]),
    stringsAsFactors = FALSE
  )
}

clean_binary_groups <- function(pred_1, pred_0) {
  pred_1 <- as.numeric(pred_1)
  pred_0 <- as.numeric(pred_0)
  list(pred_1 = pred_1[is.finite(pred_1)], pred_0 = pred_0[is.finite(pred_0)])
}

safe_auc_from_groups <- function(pred_1, pred_0) {
  x <- clean_binary_groups(pred_1, pred_0)
  if (length(x$pred_1) == 0L || length(x$pred_0) == 0L) return(NA_real_)
  actual <- c(rep(1, length(x$pred_1)), rep(0, length(x$pred_0)))
  pred <- c(x$pred_1, x$pred_0)
  as.numeric(pROC::roc(actual, pred, levels = c(0, 1), direction = "<", quiet = TRUE)$auc)
}

safe_best_threshold <- function(pred_1, pred_0) {
  x <- clean_binary_groups(pred_1, pred_0)
  if (length(x$pred_1) == 0L || length(x$pred_0) == 0L) return(NA_real_)
  actual <- c(rep(1, length(x$pred_1)), rep(0, length(x$pred_0)))
  pred <- c(x$pred_1, x$pred_0)
  roc_object <- pROC::roc(actual, pred, levels = c(0, 1), direction = "<", quiet = TRUE)
  threshold <- pROC::coords(
    roc_object, x = "best", best.method = "youden",
    ret = "threshold", transpose = FALSE
  )
  threshold <- as.numeric(unlist(threshold, use.names = FALSE))
  threshold <- threshold[!is.na(threshold)]
  if (length(threshold) == 0L) NA_real_ else min(threshold)
}

safe_threshold_indicators <- function(pred_1, pred_0, threshold) {
  x <- clean_binary_groups(pred_1, pred_0)
  if (is.na(threshold) || length(x$pred_1) == 0L || length(x$pred_0) == 0L) {
    return(c(TSS = NA_real_, kappa = NA_real_, f1 = NA_real_))
  }
  actual <- c(rep(1L, length(x$pred_1)), rep(0L, length(x$pred_0)))
  predicted <- as.integer(c(x$pred_1, x$pred_0) >= threshold)
  tp <- sum(predicted == 1L & actual == 1L)
  tn <- sum(predicted == 0L & actual == 0L)
  fp <- sum(predicted == 1L & actual == 0L)
  fn <- sum(predicted == 0L & actual == 1L)
  sensitivity <- if ((tp + fn) > 0L) tp / (tp + fn) else NA_real_
  specificity <- if ((tn + fp) > 0L) tn / (tn + fp) else NA_real_
  precision <- if ((tp + fp) > 0L) tp / (tp + fp) else NA_real_
  tss <- sensitivity + specificity - 1
  f1 <- if (is.finite(precision) && is.finite(sensitivity) &&
            (precision + sensitivity) > 0) {
    2 * precision * sensitivity / (precision + sensitivity)
  } else NA_real_
  n <- length(actual)
  p0 <- (tp + tn) / n
  pe <- ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / n^2
  kappa <- if (is.finite(pe) && pe < 1) (p0 - pe) / (1 - pe) else NA_real_
  c(TSS = tss, kappa = kappa, f1 = f1)
}

extract_model_predictions <- function(
    rst, train_presence_xy, train_background_xy,
    val_presence_xy, val_background_xy) {
  list(
    train_pred_1 = as.numeric(raster::extract(rst, train_presence_xy)),
    train_pred_0 = as.numeric(raster::extract(rst, train_background_xy)),
    val_pred_1 = as.numeric(raster::extract(rst, val_presence_xy)),
    val_pred_0 = as.numeric(raster::extract(rst, val_background_xy))
  )
}

pool_field <- function(x, field) unlist(lapply(x, `[[`, field), use.names = FALSE)

write_binary_model_prediction <- function(
    model_name, species, date, rst, threshold, presence_xy,
    binary_h5_root, binary_png_root, extent_binary, run_id,
    write_h5 = TRUE, write_png = TRUE) {
  if (is.na(threshold)) return(invisible(FALSE))
  log_binary(
    dir_run_id_h5_binary = file.path(binary_h5_root, model_name),
    dir_run_id_png_binary = file.path(binary_png_root, model_name),
    species = species,
    date = date,
    rst = rst,
    threshold = threshold,
    extent_binary = extent_binary,
    log_info = paste0(model_name, "_constantthreshold"),
    timelog = run_id,
    p = presence_xy,
    file_name = sprintf("%s_%s.h5", species, model_name),
    h5 = isTRUE(write_h5),
    png = isTRUE(write_png)
  )
  invisible(TRUE)
}
