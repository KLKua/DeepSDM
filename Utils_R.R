create_folder <- function(dir) {
  # Create a folder if it does not already exist
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
}

predict_maxent <- function(env, xm) {
  # Predict a given Maxent model 'xm' on an environmental raster stack 'env'
  result <- try(predict(env, xm, progress = ""), silent = TRUE)
  return(result)
}

plot_result <- function(sp, xm, extent_binary, p, log_info, dir_timelog_png_sp, dir_timelog_h5_sp, timelog, date = NULL) {
  # Generate and save a plot of Maxent predictions multiplied by a binary mask
  spdate <- if (is.null(date)) sp else sprintf("%s_%s", sp, date)
  date <- if (is.null(date)) "all" else date

  png(
    file.path(dir_timelog_png_sp, sprintf("%s_%s_%s.png", spdate, log_info, timelog)),
    width = 500,
    height = 1000
  )

  plot(
    xm * extent_binary,
    main = sprintf("%s_%s_%s", spdate, log_info, timelog),
    axes = FALSE,
    box = FALSE,
    legend = FALSE,
    cex.main = 0.7,
    col = color,
    breaks = seq(0, 1, 0.125)
  )
  points(p, pch = 16, col = "red", cex = 1)
  dev.off()

  # Write prediction data to an HDF5 file
  h5_file_path <- file.path(dir_timelog_h5_sp, sprintf("%s.h5", sp))
  h5_file <- H5File$new(h5_file_path, mode = "a")
  if (date %in% h5_file$ls()$name) {
    h5_file$link_delete(date)
  }
  h5attr(h5_file, "crs") <- as.character(raster::crs(extent_binary))
  extent_vals <- extent(extent_binary)
  xres <- res(extent_binary)[1]
  yres <- -res(extent_binary)[2]
  transform_values <- c(extent_vals@xmin, xres, 0, extent_vals@ymax, 0, yres)
  h5attr(h5_file, "transform") <- transform_values
  h5_file[[date]] <- t(as.matrix(xm * extent_binary))
  h5_file$close()
}

plot_result_deepsdm <- function(spdate, xm, extent_binary, p, log_info, dir_timelog_png_sp, timelog) {
  # Generate and save a plot of DeepSDM predictions multiplied by a binary mask
  png(
    file.path(dir_timelog_png_sp, sprintf("%s_%s_%s.png", spdate, log_info, timelog)),
    width = 500,
    height = 1000
  )
  plot(
    xm * extent_binary,
    main = sprintf("%s_%s_%s", spdate, log_info, timelog),
    axes = FALSE,
    box = FALSE,
    legend = FALSE,
    cex.main = 0.7,
    col = color,
    breaks = seq(0, 1, 0.125)
  )
  points(p, pch = 16, col = "red", cex = 1)
  dev.off()
}

calculate_roc <- function(px, p, bg) {
  # Calculate AUC of ROC using presence (p) and background (bg) points
  if (nrow(p) == 0 || nrow(bg) == 0) {
    return(-9999)
  } else {
    pred_1 <- raster::extract(px, p)
    pred_0 <- raster::extract(px, bg)
    actual_1 <- rep(1, nrow(p))
    actual_0 <- rep(0, nrow(bg))
    roc_obj <- roc(c(actual_1, actual_0), c(pred_1, pred_0))
    return(roc_obj$auc[1])
  }
}

calculate_indicator <- function(rst) {
  # Calculate TSS, Kappa, F1, and best threshold for training/validation presence-absence data
  if (
    nrow(xy_p_month_trainsplit) == 0 ||
    nrow(xy_p_month_valsplit) == 0 ||
    nrow(xy_pa_month_sample_trainsplit) == 0 ||
    nrow(xy_pa_month_sample_valsplit) == 0
  ) {
    return(c(-9999, -9999, -9999, -9999))
  }

  train_pred_1 <- raster::extract(rst, xy_p_month_trainsplit) %>% as.numeric()
  train_pred_0 <- raster::extract(rst, xy_pa_month_sample_trainsplit)
  train_actual_1 <- rep(1, nrow(xy_p_month_trainsplit))
  train_actual_0 <- rep(0, nrow(xy_pa_month_sample_trainsplit))
  roc_obj_train <- roc(c(train_actual_1, train_actual_0), c(train_pred_1, train_pred_0))
  best_threshold_train <- coords(roc_obj_train, "best", ret = c("threshold")) %>% pull() %>% min()

  val_pred_1 <- raster::extract(rst, xy_p_month_valsplit)
  val_pred_0 <- raster::extract(rst, xy_pa_month_sample_valsplit)
  val_actual_1 <- rep(1, nrow(xy_p_month_valsplit))
  val_actual_0 <- rep(0, nrow(xy_pa_month_sample_valsplit))
  val_predicted_classes <- ifelse(c(val_pred_1, val_pred_0) >= best_threshold_train, 1, 0)
  predicted_factor <- factor(val_predicted_classes, levels = c(0, 1))
  actual_factor <- factor(c(val_actual_1, val_actual_0), levels = c(0, 1))
  confusion_matrix <- table(predicted_factor, actual_factor)

  TP <- confusion_matrix[2, 2]
  TN <- confusion_matrix[1, 1]
  FP <- confusion_matrix[2, 1]
  FN <- confusion_matrix[1, 2]
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- TP / (TP + FP)
  N <- length(val_pred_0) + length(val_pred_1)
  p_0 <- (TP + TN) / N
  p_e <- ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / (N^2)
  TSS <- sensitivity + specificity - 1
  f1 <- (2 * precision * sensitivity) / (precision + sensitivity)
  kappa <- (p_0 - p_e) / (1 - p_e)
  return(c(TSS, kappa, f1, best_threshold_train))
}

load_env_month <- function(env_list, env_info, date, DeepSDM_conf) {
  # Load and optionally normalize raster layers for a given date
  files_env <- c()
  i <- 1
  for (env in env_list) {
    files_env[i] <- file.path(env_info$info[[env]][[date]]$tif_span_avg)
    i <- i + 1
  }
  env_month <- raster::stack(files_env)
  names(env_month) <- env_list
  for (env in env_list) {
    if (!(env %in% DeepSDM_conf$training_conf$non_normalize_env_list)) {
      values(env_month[[env]]) <- (values(env_month[[env]]) - env_info$info[[env]]$mean) / env_info$info[[env]]$sd
    }
  }
  return(env_month)
}

load_env_allmonth <- function(env_list, env_info, date_list_all_selectmonth, DeepSDM_conf) {
  # Load multiple month layers, normalize them if needed, and compute the mean across all specified dates
  date_list_all_selectmonth <- as.vector(date_list_all_selectmonth)
  env_allmonth_list <- list()

  for (date in date_list_all_selectmonth) {
    files_env <- lapply(env_list, function(env) {
      file.path(env_info$info[[env]][[date]]$tif_span_avg)
    }) %>% unlist()
    env_allmonth <- raster::stack(files_env)
    names(env_allmonth) <- env_list
    lapply(env_list, function(env) {
      if (!(env %in% DeepSDM_conf$training_conf$non_normalize_env_list)) {
        values(env_allmonth[[env]]) <<- (values(env_allmonth[[env]]) - env_info$info[[env]]$mean) / env_info$info[[env]]$sd
      }
    })
    env_allmonth_list[[date]] <- env_allmonth
  }

  layer_means <- lapply(seq_along(env_list), function(layer_index) {
    print(paste0("env_", layer_index))
    layer_stack <- stack(lapply(names(env_allmonth_list), function(t) {
      raster::raster(env_allmonth_list[[t]], layer_index)
    }))
    calc(layer_stack, fun = mean)
  })
  out <- raster::stack(layer_means)
  names(out) <- env_list
  return(out)
}

set_default_variable <- function(default_value = -9999) {
  # Reset various global variables used to store metrics and sample counts
  deepsdm_all_month_val <<- default_value
  deepsdm_all_month_train <<- default_value
  deepsdm_all_month_all <<- default_value
  maxent_month_month_all <<- default_value
  maxent_month_month_train <<- default_value
  maxent_month_month_val <<- default_value
  maxent_all_month_val <<- default_value
  maxent_all_month_train <<- default_value
  maxent_all_month_all <<- default_value
  maxent_TSS <<- default_value
  deepsdm_TSS <<- default_value
  maxent_kappa <<- default_value
  deepsdm_kappa <<- default_value
  maxent_f1 <<- default_value
  deepsdm_f1 <<- default_value
  maxent_threshold <<- default_value
  deepsdm_threshold <<- default_value
  p_month <<- default_value
  p_valpart_month <<- default_value
  p_trainpart_month <<- default_value
  pa_valpart_month <<- default_value
  pa_trainpart_month <<- default_value

  p_month <<- ifelse(exists("xy_p_month"), nrow(xy_p_month), default_value)
  p_valpart_month <<- ifelse(exists("xy_p_month_valsplit"), nrow(xy_p_month_valsplit), default_value)
  p_trainpart_month <<- ifelse(exists("xy_p_month_trainsplit"), nrow(xy_p_month_trainsplit), default_value)
  pa_valpart_month <<- ifelse(exists("xy_pa_month_sample_valsplit"), nrow(xy_p_month_trainsplit), default_value)
  pa_trainpart_month <<- ifelse(exists("xy_pa_month_sample_trainsplit"), nrow(xy_p_month_trainsplit), default_value)
}

set_default_variable_all <- function(default_value = -9999) {
  # Reset various global variables used to store "all-month" metrics and sample counts
  maxent_all_all_val <<- default_value
  maxent_all_all_train <<- default_value
  maxent_all_all_all <<- default_value
  deepsdm_all_all_val <<- default_value
  deepsdm_all_all_train <<- default_value
  deepsdm_all_all_all <<- default_value
  p_all <<- default_value
  p_valpart_all <<- default_value
  p_trainpart_all <<- default_value
  pa_valpart_all <<- default_value
  pa_trainpart_all <<- default_value

  p_all <<- ifelse(exists("xy_p_all"), nrow(xy_p_all), default_value)
  p_valpart_all <<- ifelse(exists("xy_p_all_valsplit"), nrow(xy_p_all_valsplit), default_value)
  p_trainpart_all <<- ifelse(exists("xy_p_all_trainsplit"), nrow(xy_p_all_trainsplit), default_value)
  pa_valpart_all <<- ifelse(exists("xy_pa_all_sample_valsplit"), nrow(xy_pa_all_sample_valsplit), default_value)
  pa_trainpart_all <<- ifelse(exists("xy_pa_all_sample_trainsplit"), nrow(xy_pa_all_sample_trainsplit), default_value)
}

generate_points <- function(num_pa = 10000) {
  # Generate month presence and pseudo-absence points
  occ_rst_path <- file.path(".", sp_info$dir_base, sp_info$file_name[[species]][[date]])
  occ_rst <- raster::raster(occ_rst_path)
  i_p_occ_rst <- which(values(occ_rst) == 1)
  i_pa_occ_rst <- which(values(occ_rst) == 0)
  xy_p_month <<- xyFromCell(occ_rst, i_p_occ_rst)
  xy_p_month_trainsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_trainsplit))
  xy_p_month_valsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_valsplit))
  i_pa_month <- intersect(i_pa_occ_rst, i_extent)

  if (num_pa == "num_p") {
    i_pa_month_sample <- sample(i_pa_month, nrow(xy_p_month))
  } else {
    i_pa_month_sample <- sample(i_pa_month, num_pa)
  }

  xy_pa_month_sample <<- xyFromCell(occ_rst, i_pa_month_sample)
  xy_pa_month_sample_trainsplit <<- xyFromCell(occ_rst, intersect(i_pa_month_sample, i_trainsplit))
  xy_pa_month_sample_valsplit <<- xyFromCell(occ_rst, intersect(i_pa_month_sample, i_valsplit))
}

generate_points_all <- function(date_list_all) {
  # Combine presence from multiple rasters and generate a common pseudo-absence set
  occ_rst_path <- lapply(date_list_all, function(date) {
    file.path(".", sp_info$dir_base, sp_info$file_name[[species]][[date]])
  }) %>% unlist()

  occ_rst <- raster::stack(occ_rst_path)
  occ_rst <- calc(occ_rst, function(x) sign(sum(x, na.rm = TRUE)))

  i_p_occ_rst <- which(values(occ_rst) == 1)
  i_pa_occ_rst <- which(values(occ_rst) == 0)
  xy_p_all <<- xyFromCell(occ_rst, i_p_occ_rst)
  xy_p_all_trainsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_trainsplit))
  xy_p_all_valsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_valsplit))

  i_pa_all <- intersect(i_pa_occ_rst, i_extent)
  i_pa_all_sample <- sample(i_pa_all, 10000)
  xy_pa_all_sample <<- xyFromCell(occ_rst, i_pa_all_sample)
  xy_pa_all_sample_trainsplit <<- xyFromCell(occ_rst, intersect(i_pa_all_sample, i_trainsplit))
  xy_pa_all_sample_valsplit <<- xyFromCell(occ_rst, intersect(i_pa_all_sample, i_valsplit))
}

check_dataset_in_h5 <- function(h5_path, dataset_name) {
  # Check if an HDF5 file contains a specified dataset
  if (file.exists(h5_path)) {
    h5_file <- H5File$new(h5_path, mode = "r")
    dataset_in_h5 <- dataset_name %in% h5_file$ls()$name
    h5_file$close()
    return(dataset_in_h5)
  } else {
    return(FALSE)
  }
}

h5dataset_to_raster <- function(h5_path, dataset_name) {
  # Load an HDF5 dataset and convert it into a RasterLayer
  h5_file <- H5File$new(h5_path, mode = "r")
  crs_val <- h5attributes(h5_file)$crs
  transform <- h5attributes(h5_file)$transform
  h5_array <- h5_file[[dataset_name]]

  extent_data <- extent(
    transform[1],
    transform[1] + transform[2] * h5_array$dims[1],
    transform[4] + transform[6] * h5_array$dims[2],
    transform[4]
  )

  crs_data <- CRS(crs_val)
  raster_output <- raster::raster(
    t(h5_array[1:h5_array$dims[1], 1:h5_array$dims[2]])
  )
  extent(raster_output) <- extent_data
  crs(raster_output) <- crs_data
  h5_file$close()
  return(raster_output)
}

log_binary <- function(
  dir_run_id_h5_binary,
  dir_run_id_png_binary,
  species,
  date,
  rst,
  threshold,
  extent_binary,
  log_info,
  timelog,
  p,
  file_name,
  h5 = TRUE,
  png = TRUE
) {
  # Convert predictions to binary using a threshold and save to HDF5 and/or PNG
  rst[rst >= threshold] <- 1
  rst[rst < threshold] <- 0

  if (h5) {
    dir_run_id_h5_binary_sp <- file.path(dir_run_id_h5_binary, species)
    create_folder(dir_run_id_h5_binary_sp)
    h5_file_path <- file.path(dir_run_id_h5_binary_sp, file_name)
    h5_file <- H5File$new(h5_file_path, mode = "a")

    if (date %in% h5_file$ls()$name) {
      h5_file[[date]]$delete()
    }
    h5attr(h5_file, "crs") <- as.character(raster::crs(extent_binary))
    extent_vals <- extent(extent_binary)
    xres <- res(extent_binary)[1]
    yres <- -res(extent_binary)[2]
    transform_values <- c(extent_vals@xmin, xres, 0, extent_vals@ymax, 0, yres)
    h5attr(h5_file, "transform") <- transform_values

    h5_file[[date]] <- t(as.matrix(rst * extent_binary))
    h5_file$close()
  }

  if (png) {
    dir_run_id_png_binary_sp <- file.path(dir_run_id_png_binary, species)
    create_folder(dir_run_id_png_binary_sp)
    png(
      file.path(dir_run_id_png_binary_sp, sprintf("%s_%s_%s_%s.png", species, date, log_info, timelog)),
      width = 500,
      height = 1000
    )
    plot(
      rst * extent_binary,
      main = sprintf("%s_%s_%s_%s", species, date, log_info, timelog),
      axes = FALSE,
      box = FALSE,
      legend = FALSE,
      cex.main = 0.7,
      col = color,
      breaks = seq(0, 1, 0.125)
    )
    points(p, pch = 16, col = "black", cex = 1)
    dev.off()
  }
}

calculate_thresholddepend_indi <- function(pred_1, pred_0, actual_1, actual_0, threshold) {
  # Compute TSS, Kappa, and F1 given a threshold for classification
  pred_classes <- ifelse(c(pred_1, pred_0) >= threshold, 1, 0)
  pred_factor <- factor(pred_classes, levels = c(0, 1))
  actual_factor <- factor(c(actual_1, actual_0), levels = c(0, 1))
  confusion_matrix <- table(pred_factor, actual_factor)
  TP <- confusion_matrix[2, 2]
  TN <- confusion_matrix[1, 1]
  FP <- confusion_matrix[2, 1]
  FN <- confusion_matrix[1, 2]
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- TP / (TP + FP)
  N <- length(pred_0) + length(pred_1)
  p_0 <- (TP + TN) / N
  p_e <- ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / (N^2)
  TSS <- sensitivity + specificity - 1
  f1 <- (2 * precision * sensitivity) / (precision + sensitivity)
  kappa <- (p_0 - p_e) / (1 - p_e)
  return(c(TSS, kappa, f1))
}