library(hdf5r)                                  # Provide read/write access to HDF5 files
library(raster)                                 # For raster data handling
library(dismo)                                  # For species distribution modeling
library(pROC)                                   # For calculating ROC/AUC
library(tidyverse)                              # Data manipulation
library(rjson)                                  # Parsing JSON files
library(yaml)                                   # Parsing YAML files
library(rJava)                                  # Java interface
set.seed(42)                                    # Fix random seed

source("Utils_R.R")                    # Source custom functions (utilities)

args = commandArgs(trailingOnly = TRUE)         # Capture command-line arguments
if (length(args) == 0) {
  stop("At least one argument must be supplied (input file).n", call. = FALSE)
}

run_id <- "e52c8ac9a3e24c75ac871f63bbdea060"     # Unique identifier for this run
exp_id <- "115656750127464383"                  # Experiment ID

# Prepare base directories for the current run
dir_base_run_id <- file.path("predicts_maxent", run_id)
dir_run_id_png <- file.path(dir_base_run_id, "png", "all")
create_folder(dir_run_id_png)
dir_run_id_h5 <- file.path(dir_base_run_id, "h5", "all")
create_folder(dir_run_id_h5)
dir_run_id_png_binary <- file.path(dir_base_run_id, "png", "binary")
create_folder(dir_run_id_png_binary)
dir_run_id_h5_binary <- file.path(dir_base_run_id, "h5", "binary")
create_folder(dir_run_id_h5_binary)
dir_run_id_env_contribution <- file.path(dir_base_run_id, "env_contribution", "all")
create_folder(dir_run_id_env_contribution)
dir_maxent_model <- file.path(dir_base_run_id, "maxent_model")
create_folder(dir_maxent_model)

# Load configuration information for DeepSDM
DeepSDM_conf_path <- file.path("predicts", run_id, "DeepSDM_conf.yaml")
DeepSDM_conf <- yaml.load_file(DeepSDM_conf_path)
env_list <- sort(DeepSDM_conf$training_conf$env_list)   # Sorted list of environmental variables

# Load geographic mask and partition info
extent_binary <- raster(DeepSDM_conf$geo_extent_file)   # 1 for land, 0 for sea
trainval_split <- raster(file.path("mlruns", exp_id, run_id, "artifacts", "extent_binary", "partition_extent.tif"))
i_extent <- which(values(extent_binary) == 1)           # Indices of land area
i_trainsplit <- which(values(trainval_split) == 1)      # Indices for training partition
i_valsplit <- which(is.na(values(trainval_split)))      # Indices for validation partition

# Load environmental info and species info
env_info_path <- file.path("predicts", run_id, "env_inf.json")
env_info <- fromJSON(file = env_info_path)
sp_info_path <- file.path("predicts", run_id, "sp_inf.json")
sp_info <- fromJSON(file = sp_info_path)

# Lists of dates for prediction and training
date_list_predict <- DeepSDM_conf$training_conf$date_list_predict
date_list_train <- DeepSDM_conf$training_conf$date_list_train

# Sorted list of species for training/prediction
species_list <- sort(DeepSDM_conf$training_conf$species_list_predict)

# Data frames to store results across months
df_all_month <- data.frame(
  spdate = character(),
  maxent_all_month_val = numeric(),
  maxent_all_month_train = numeric(),
  maxent_all_month_all = numeric(),
  deepsdm_all_month_val = numeric(),
  deepsdm_all_month_train = numeric(),
  deepsdm_all_month_all = numeric(),
  maxent_TSS = numeric(),
  deepsdm_TSS = numeric(),
  maxent_kappa = numeric(),
  deepsdm_kappa = numeric(),
  maxent_f1 = numeric(),
  deepsdm_f1 = numeric(),
  p_month = numeric(),
  p_valpart_month = numeric(),
  p_trainpart_month = numeric(),
  pa_valpart_month = numeric(),
  pa_trainpart_month = numeric(),
  maxent_threshold = numeric(),
  deepsdm_threshold = numeric()
)

# Data frame to store overall "all-month" results
df_all_all <- data.frame(
  species = character(),
  maxent_all_all_val = numeric(),
  maxent_all_all_train = numeric(),
  maxent_all_all_all = numeric(),
  p_all = numeric(),
  p_valpart_all = numeric(),
  p_trainpart_all = numeric(),
  pa_valpart_all = numeric(),
  pa_trainpart_all = numeric()
)

# Color scale for plotting
color <- c("#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704")

# Load and preprocess environmental layers for training
env_all <- load_env_allmonth(env_list, env_info, date_list_train, DeepSDM_conf)

# Read the starting index from command line
r_start <- as.numeric(args[1])
r_end <- r_start + 2

# Loop through subset of species from r_start to r_end
for (species in species_list[r_start:min(r_end, length(species_list))]) {
  # Create directories for outputs specific to this species
  dir_run_id_png_sp <- file.path(dir_run_id_png, species)
  create_folder(dir_run_id_png_sp)
  dir_run_id_h5_sp <- file.path(dir_run_id_h5, species)
  create_folder(dir_run_id_h5_sp)

  # Try to generate presence/absence data for all training dates
  result <- tryCatch({
    generate_points_all(date_list_train)
    TRUE
  }, error = function(e) {
    message("Error encountered: ", e$message)
    FALSE
  })

  if (!result) {
    message("Skipping this iteration due to an error.")
    next
  }

  # Reset global variables for storing metrics
  set_default_variable_all()

  # Check if "all" predictions exist for this species
  maxent_all_all_exists <- FALSE
  maxent_h5_path <- file.path(dir_run_id_h5_sp, sprintf("%s.h5", species))
  maxent_all_all_exists <- check_dataset_in_h5(maxent_h5_path, "all")

  if (!maxent_all_all_exists) {
    # Train a Maxent model using all-month data
    xm_all <- try(maxent(x = env_all, p = xy_p_all_trainsplit, a = xy_pa_all_sample_trainsplit), silent = TRUE)
    save(xm_all, file = file.path(dir_maxent_model, sprintf("%s_all.RData", species)))
    if (!is.character(xm_all)) {
      write.csv(
        xm_all@results,
        file.path(dir_run_id_env_contribution, sprintf("%s_env_contribution_maxentall.csv", species))
      )
      # Generate predictions for the entire region
      maxent_all_all <- predict_maxent(env_all, xm_all)
      maxent_all_all_exists <- TRUE

      # Save plot & data
      plot_result(
        species,
        maxent_all_all,
        extent_binary,
        xy_p_all,
        "maxent_all_all",
        dir_run_id_png_sp,
        dir_run_id_h5_sp,
        run_id
      )

      # Compute AUC metrics
      maxent_all_all_train <- calculate_roc(maxent_all_all, xy_p_all_trainsplit, xy_pa_all_sample_trainsplit)
      maxent_all_all_val <- calculate_roc(maxent_all_all, xy_p_all_valsplit, xy_pa_all_sample_valsplit)
      maxent_all_all_all <- calculate_roc(maxent_all_all, xy_p_all, xy_pa_all_sample)
    }
  } else {
    # If already exists, load from HDF5
    maxent_all_all <- h5dataset_to_raster(maxent_h5_path, "all")
    maxent_all_all_exists <- TRUE

    # Load saved model object
    load(file.path(dir_maxent_model, sprintf("%s_all.RData", species)))

    # Compute AUC metrics
    maxent_all_all_train <- calculate_roc(maxent_all_all, xy_p_all_trainsplit, xy_pa_all_sample_trainsplit)
    maxent_all_all_val <- calculate_roc(maxent_all_all, xy_p_all_valsplit, xy_pa_all_sample_valsplit)
    maxent_all_all_all <- calculate_roc(maxent_all_all, xy_p_all, xy_pa_all_sample)
  }

  # Store "all-month" metrics
  df_all_all[nrow(df_all_all) + 1, ] <- c(
    species,
    maxent_all_all_val,
    maxent_all_all_train,
    maxent_all_all_all,
    p_all,
    p_valpart_all,
    p_trainpart_all,
    pa_valpart_all,
    pa_trainpart_all
  )

  # Loop through all dates to generate predictions
  for (date in date_list_predict) {
    print(paste("start", species, date))
    sp_month <- paste0(species, "_", date)
    set_default_variable()

    maxent_all_month_exists <- FALSE
    if (maxent_all_all_exists) {
      # Generate presence/absence for this date
      generate_points(num_pa = "num_p")
      set_default_variable()

      # Load environment for the specific date
      env_month <- load_env_month(env_list, env_info, date, DeepSDM_conf)

      # Check if predictions for this date exist in the HDF5
      maxent_all_month_exists <- check_dataset_in_h5(maxent_h5_path, date)
      if (maxent_all_month_exists) {
        maxent_all_month <- h5dataset_to_raster(maxent_h5_path, date)
      } else {
        # Predict with the previously trained model
        maxent_all_month <- predict_maxent(env_month, xm_all)
        maxent_all_month_exists <- TRUE
        plot_result(
          species,
          maxent_all_month,
          extent_binary,
          xy_p_month,
          "maxent_all_month",
          dir_run_id_png_sp,
          dir_run_id_h5_sp,
          run_id,
          date
        )
      }

      # Compute AUC metrics
      maxent_all_month_train <- calculate_roc(maxent_all_month, xy_p_month_trainsplit, xy_pa_month_sample_trainsplit)
      maxent_all_month_val <- calculate_roc(maxent_all_month, xy_p_month_valsplit, xy_pa_month_sample_valsplit)
      maxent_all_month_all <- calculate_roc(maxent_all_month, xy_p_month, xy_pa_month_sample)

      # Compute TSS, kappa, F1, threshold
      maxent_other_indicator <- calculate_indicator(maxent_all_month)
      maxent_TSS <- maxent_other_indicator[1]
      maxent_kappa <- maxent_other_indicator[2]
      maxent_f1 <- maxent_other_indicator[3]
      maxent_threshold <- maxent_other_indicator[4]

      # Check if DeepSDM predictions exist for this date
      deepsdm_h5_path <- file.path("predicts", run_id, "h5", species, sprintf("%s.h5", species))
      deepsdm_all_month_exists <- check_dataset_in_h5(deepsdm_h5_path, date)
      if (deepsdm_all_month_exists) {
        deepsdm_all_month <- h5dataset_to_raster(deepsdm_h5_path, date)
        plot_result_deepsdm(
          sp_month,
          deepsdm_all_month,
          extent_binary,
          xy_p_month,
          "deepsdm_all_month",
          dir_run_id_png_sp,
          run_id
        )
        deepsdm_all_month_train <- try(calculate_roc(deepsdm_all_month, xy_p_month_trainsplit, xy_pa_month_sample_trainsplit))
        deepsdm_all_month_val <- try(calculate_roc(deepsdm_all_month, xy_p_month_valsplit, xy_pa_month_sample_valsplit))
        deepsdm_all_month_all <- try(calculate_roc(deepsdm_all_month, xy_p_month, xy_pa_month_sample))
        deepsdm_other_indicator <- calculate_indicator(deepsdm_all_month)
        deepsdm_TSS <- deepsdm_other_indicator[1]
        deepsdm_kappa <- deepsdm_other_indicator[2]
        deepsdm_f1 <- deepsdm_other_indicator[3]
        deepsdm_threshold <- deepsdm_other_indicator[4]
      }
    }

    # Record results in data frame
    df_all_month[nrow(df_all_month) + 1, ] <- c(
      sp_month,
      maxent_all_month_val,
      maxent_all_month_train,
      maxent_all_month_all,
      deepsdm_all_month_val,
      deepsdm_all_month_train,
      deepsdm_all_month_all,
      maxent_TSS,
      deepsdm_TSS,
      maxent_kappa,
      deepsdm_kappa,
      maxent_f1,
      deepsdm_f1,
      p_month,
      p_valpart_month,
      p_trainpart_month,
      pa_valpart_month,
      pa_trainpart_month,
      maxent_threshold,
      deepsdm_threshold
    )
  }
}

# Save month-based results to CSV
output_csv_all_month_path <- file.path(dir_base_run_id, sprintf("model_performance_diffthreshold%03d.csv", r_start))
write.csv(df_all_month, output_csv_all_month_path, row.names = FALSE)