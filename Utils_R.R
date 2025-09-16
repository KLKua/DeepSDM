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
  pa_valpart_month <<- ifelse(exists("xy_pa_month_sample_valsplit"), nrow(xy_pa_month_sample_valsplit), default_value)
  pa_trainpart_month <<- ifelse(exists("xy_pa_month_sample_trainsplit"), nrow(xy_pa_month_sample_trainsplit), default_value)
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
  # 只從 HDF5 讀該月出現棧格；沒有就報錯
  h5_fname <- sp_info$file_name[[species]]$h5file_name
  if (is.null(h5_fname))
    stop("h5file_name not found in sp_info for species: ", species)

  h5_path   <- file.path(sp_info$dir_base, h5_fname)
  dset_name <- sp_info$file_name[[species]]$h5file_dataset_name[[date]]

  if (is.null(dset_name) || !check_dataset_in_h5(h5_path, dset_name))
    stop(sprintf("HDF5 dataset missing for %s @ %s (file=%s, dataset=%s)",
                 species, date, h5_path, dset_name))

  # 用 HDF5→Raster 的工具，並以 extent_binary 當範本補齊座標資訊
  occ_rst <- h5dataset_to_raster(h5_path, dset_name, template = extent_binary)

  # 建議用 >0 視為出現（多數 HDF5 並非只用 1）
  i_p_occ_rst  <- which(raster::values(occ_rst) > 0)
  i_pa_occ_rst <- which(raster::values(occ_rst) == 0)

  xy_p_month            <<- raster::xyFromCell(occ_rst, i_p_occ_rst)
  xy_p_month_trainsplit <<- raster::xyFromCell(occ_rst, intersect(i_p_occ_rst, i_trainsplit))
  xy_p_month_valsplit   <<- raster::xyFromCell(occ_rst, intersect(i_p_occ_rst, i_valsplit))

  i_pa_month <- intersect(i_pa_occ_rst, i_extent)
  if (num_pa == "num_p") {
    i_pa_month_sample <- if (nrow(xy_p_month) > 0) sample(i_pa_month, nrow(xy_p_month)) else integer(0)
  } else {
    i_pa_month_sample <- sample(i_pa_month, num_pa)
  }
  xy_pa_month_sample            <<- raster::xyFromCell(occ_rst, i_pa_month_sample)
  xy_pa_month_sample_trainsplit <<- raster::xyFromCell(occ_rst, intersect(i_pa_month_sample, i_trainsplit))
  xy_pa_month_sample_valsplit   <<- raster::xyFromCell(occ_rst, intersect(i_pa_month_sample, i_valsplit))
}


generate_points_all <- function(date_list_all,
                                n_bg = 10000) {
  # ── 1. 找到 HDF5 檔 ────────────────────────────────────────────
  h5_fname <- sp_info$file_name[[species]]$h5file_name
  if (is.null(h5_fname))
    stop("h5file_name not found in sp_info for species: ", species)

  h5_path <- file.path(sp_info$dir_base, h5_fname)
  if (!file.exists(h5_path))
    stop("HDF5 file does not exist: ", h5_path)
  # ── 2. 逐月份讀 dataset → RasterLayer list ────────────────────
  ras_list <- lapply(date_list_all, function(dt) {
    dset_name <- sp_info$file_name[[species]]$h5file_dataset_name[[dt]]
    if (is.null(dset_name)) {
      message(sprintf("[skip] %s - dataset name missing for %s", species, dt))
      return(NULL)
    }
    if (!check_dataset_in_h5(h5_path, dset_name)) {
      message(sprintf("[skip] %s - dataset %s not in HDF5", species, dset_name))
      return(NULL)
    }
    h5dataset_to_raster(h5_path, dset_name)
  })
  ras_list <- Filter(Negate(is.null), ras_list)   # 清除 NULL
  if (length(ras_list) == 0)
    stop("No occurrence layers found in HDF5 for the given dates.")

  # ── 3. 疊成一張「是否曾出現」的單層 raster ─────────────────────
  occ_rst <- raster::stack(ras_list)
  occ_rst <- raster::calc(occ_rst, fun = function(x) sign(sum(x, na.rm = TRUE)))
  # ── 4. 依舊生成 presence / pseudo-absence 點集 ────────────────
  i_p_occ_rst  <- which(raster::values(occ_rst) == 1)
  i_pa_occ_rst <- which(raster::values(occ_rst) == 0)

  xy_p_all                 <<- raster::xyFromCell(occ_rst, i_p_occ_rst)
  xy_p_all_trainsplit      <<- raster::xyFromCell(occ_rst, intersect(i_p_occ_rst, i_trainsplit))
  xy_p_all_valsplit        <<- raster::xyFromCell(occ_rst, intersect(i_p_occ_rst, i_valsplit))

  i_pa_all                 <- intersect(i_pa_occ_rst, i_extent)
  i_pa_all_sample          <- sample(i_pa_all, n_bg)
  xy_pa_all_sample         <<- raster::xyFromCell(occ_rst, i_pa_all_sample)
  xy_pa_all_sample_trainsplit <<- raster::xyFromCell(occ_rst, intersect(i_pa_all_sample, i_trainsplit))
  xy_pa_all_sample_valsplit   <<- raster::xyFromCell(occ_rst, intersect(i_pa_all_sample, i_valsplit))
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

h5dataset_to_raster <- function(h5_path, dataset_name, template = extent_binary) {
  h5_file <- H5File$new(h5_path, mode = "r")
  crs_val  <- h5attributes(h5_file)$crs
  transform <- h5attributes(h5_file)$transform
  h5_array <- h5_file[[dataset_name]]
  mat <- t(h5_array[1:h5_array$dims[1], 1:h5_array$dims[2]])
  rst <- raster::raster(mat)

  if (!is.null(transform)) {
    extent(rst) <- extent(
      transform[1],
      transform[1] + transform[2] * h5_array$dims[1],
      transform[4] + transform[6] * h5_array$dims[2],
      transform[4]
    )
  } else if (!is.null(template)) {
    extent(rst) <- extent(template)      # 用範本
    res(rst)    <- res(template)
  } else {
    stop("HDF5 缺少 'transform' 屬性，且未提供 template。")
  }

  if (!is.null(crs_val)) {
    crs(rst) <- CRS(crs_val)
  } else if (!is.null(template)) {
    crs(rst) <- crs(template)            # 用範本
  } # 否則就維持未定義 CRS（或你也可 stop）

  h5_file$close()
  return(rst)
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

# =============================================================================
#  build_maxent_training_df()
#  ---------------------------
#  依你在 A 中的完整流程：
#    ① 讀 conf / env_info / sp_info
#    ② 把指定 date_list 的所有環境圖層 → env_all_df
#    ③ 取 presence（HDF5）資料
#    ④ 依 date quota 抽 10 000 個 background
#    ⑤ 合併成 env_final_df，並回傳：
#         - train_df   ：僅環境因子欄位（供 maxent(x=…)）
#         - p_vec      ：presence 標記 (1 / 0)
#         - pres_xy_df ：presence xy（供畫圖）
#         - abs_xy_df  ：absence xy （供 ROC）
#  備註：完全沿用 A 的變數命名與處理順序，只改寫成函式形式。
# =============================================================================
build_maxent_training_df <- function(date_list,
                                     n_bg       = 10000
                                     ) {

  # -- 0.  套用 A 段的前置 ------------------------------------------------------

  # -- 1.  載入所有日期的環境層 → env_all_df  ----------------------------------
  load_env <- function(vars, info, dates, DeepSDM_conf) {
    dplyr::bind_rows(lapply(dates, function(dt) {
      files <- sapply(vars, function(v) info$info[[v]][[dt]]$tif_span_avg)
      stk   <- raster::stack(files); names(stk) <- vars
      # 標準化（同 A）
      for (v in vars[!vars %in% DeepSDM_conf$training_conf$non_normalize_env_list]) {
        stk[[v]] <- (stk[[v]] - info$info[[v]]$mean) / info$info[[v]]$sd
      }
      raster::as.data.frame(stk, xy = TRUE, na.rm = TRUE) |>
        dplyr::mutate(date = dt)
    }))
  }
  env_all_df <- load_env(env_list, env_info, date_list, DeepSDM_conf)

  # -- 2.  讀取 presence HDF5  -------------------------------------------------
    get_presence_xy <- function(sp_info, species, template, date_list) {
      h5 <- hdf5r::H5File$new(file.path(sp_info$dir_base, sp_info$file_name[[species]]$h5file_name), "r")
      out <- lapply(as.vector(date_list), function(dt) {
        dset_name <- sp_info$file_name[[species]]$h5file_dataset_name[[dt]]
        if (is.null(dset_name) || !h5$exists(paste0('/', dset_name))) return(NULL)
        mat <- h5[[paste0('/', dset_name)]]$read()
        if (length(dim(mat)) == 2) mat <- t(mat)             # 與 h5dataset_to_raster 一致的轉置 :contentReference[oaicite:8]{index=8}
        ras <- raster::raster(template); raster::values(ras) <- as.vector(mat)
        cells <- raster::Which(ras > 0, cells = TRUE)        # 用 >0 避免只抓 ==1 的侷限 :contentReference[oaicite:9]{index=9}
        if (!length(cells)) return(NULL)
        xy <- raster::xyFromCell(template, cells)
        data.frame(x = xy[,1], y = xy[,2], date = dt)        # 用 date_list 的字串，與環境表一致 :contentReference[oaicite:10]{index=10}
      })
      h5$close()
      dplyr::bind_rows(out) |> dplyr::distinct()
    }


  # template <- raster::raster(env_info$info[[env_list[1]]][[date_list[1]]]$tif_span_avg)
  template <- extent_binary
  presence_xy_df  <- get_presence_xy(sp_info, species, template, date_list)

  env_presence_df <- dplyr::inner_join(env_all_df, presence_xy_df,
                                       by = c("x", "y", "date")) |>
                     dplyr::mutate(presence = 1)

  # -- 3.  抽背景點（完全照 A 的 quota 寫法） ----------------------------------
  absence_pool <- dplyr::anti_join(env_all_df, presence_xy_df,
                                   by = c("x", "y", "date"))
  if (nrow(absence_pool) == 0)
    stop("No background cells available.")

  dates <- sort(unique(absence_pool$date))
  n_dates <- length(dates)
  quota <- rep(floor(n_bg / n_dates), n_dates)
  if (n_bg %% n_dates)
    quota[seq_len(n_bg %% n_dates)] <- quota[seq_len(n_bg %% n_dates)] + 1
  names(quota) <- dates

  abs_list <- vector("list", n_dates); missing <- 0L
  for (i in seq_along(dates)) {
    dt   <- dates[i]; q <- quota[i]
    pool <- dplyr::filter(absence_pool, date == dt)
    if (nrow(pool) == 0) {
      missing <- missing + q; next
    }
    abs_list[[i]] <- dplyr::slice_sample(pool, n = q,
                                         replace = nrow(pool) < q)
  }
  env_absence_df <- dplyr::bind_rows(abs_list)
  if (nrow(env_absence_df) < n_bg) {
    extra <- n_bg - nrow(env_absence_df)
    donor_pool <- dplyr::filter(absence_pool,
                                date == dates[which(lengths(abs_list) > 0)[1]])
    env_absence_df <- dplyr::bind_rows(
      env_absence_df,
      dplyr::slice_sample(donor_pool, n = extra, replace = TRUE)
    )
  }
  env_absence_df <- dplyr::mutate(env_absence_df, presence = 0)

  # -- 4.  合併並輸出 ---------------------------------------------------------
  env_final_df <- dplyr::bind_rows(env_presence_df, env_absence_df)

  message(sprintf("Presence rows : %d", sum(env_final_df$presence == 1)))
  message(sprintf("Absence rows  : %d", sum(env_final_df$presence == 0)))

  # x（predictor）只留環境因子欄位
  train_df <- env_final_df[, env_list, drop = FALSE]
  p_vec    <- env_final_df$presence

  return(list(
    train_df       = train_df,
    p_vec          = p_vec,
    pres_xy_df     = presence_xy_df[, c("x", "y")],
    abs_xy_df      = env_absence_df[, c("x", "y")],
    env_list       = env_list
  ))
}
# =============================================================================
