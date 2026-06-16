# SEAM-SDM Project Guide

Species‑Embedding & Attention Multiscale U‑Net for Species Distribution Modeling (SEAM-SDM) predicts monthly habitat suitability by combining species co-occurrence embeddings, multi-scale environmental rasters, and survey-effort weighting. This repository hosts every script, notebook, and helper module required to reproduce the manuscript pipeline—from raw data harmonization to DeepSDM training, MaxEnt baselines, and figure generation.

## Table of Contents
- [Environment Setup](#environment-setup)
- [External Artifacts on Google Drive](#external-artifacts-on-google-drive)
- [Repository Layout](#repository-layout)
- [Execution Flow](#execution-flow)
- [Script and Module Reference](#script-and-module-reference)
  - [Python](#python)
  - [R](#r)
  - [Shell](#shell)
- [Primary Outputs](#primary-outputs)
- [Additional Resources](#additional-resources)

## Environment Setup

### Python
1. Create a Conda environment on a CUDA-capable machine.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. To exactly match the original experiments, build the Conda environment specified in `python_env.yaml`.

### R
1. Install the packages listed in `requirements_r.txt`, then follow `setup_r_environment.md` to prepare the MaxEnt toolchain.

## External Artifacts on Google Drive
All pretrained checkpoints, curated datasets, intermediate logs, predictions, and figure assets reside on Google Drive: <https://drive.google.com/drive/folders/1zzJg_q1gTyvoprR7r4iX69xrOYRsJlGR?usp=drive_link>.

Download every archive in that folder and extract it **inside** this repository so the resulting directories sit under `DeepSDM/` (for example: `/path/to/projects/DeepSDM`, `/path/to/projects/DeepSDM/workspace`, `/path/to/projects/DeepSDM/raw`, ...). Configuration files and notebooks assume this relative layout when resolving paths.

### Archive contents
- **`workspace.zip` → `workspace/`** – cached species metadata, occurrence grids, survey-effort rasters, and partition masks consumed by the Lightning data modules during training and evaluation.
- **`medium.zip` → `medium/`** – medium-resolution environmental mosaics produced by the preprocessing notebook and referenced when rebuilding caches.
- **`mlruns.zip` → `mlruns/`** – MLflow runs with DeepSDM checkpoints, configuration snapshots, and validation metrics that the prediction notebook and R scripts reuse.
- **`predicts.zip` → `predicts/`** – DeepSDM prediction rasters (HDF5/PNG) plus auxiliary JSON summaries produced by inference.
- **`predicts_maxent.zip` → `predicts_maxent/`** – MaxEnt prediction rasters, binary masks, and evaluation tables used during threshold and baseline comparisons.
- **`plots.zip` → `plots/`** – curated figure assets and intermediate tables for the manuscript and supplementary notebooks.
- **`raw.z.001`, `raw.z.002`, `raw.z.003` → `raw/`** – multi-part archive containing the raw climate, land-cover, and occurrence sources required by `01_prepare_data.ipynb`. Combine the pieces to materialize the `raw/` directory alongside the repository before running data preparation.

## Repository Layout

| Category | Key Files | Description |
| --- | --- | --- |
| Configuration | `DeepSDM_conf.yaml` | Centralizes species lists, temporal ranges, environmental variables, cache directories, and Lightning trainer options. |
| Lightning modules | `LitDeepSDMData.py`, `LitUNetSDM.py`, `LitUNetSDM_prediction.py` | Data modules and training/prediction loops that wrap the U-Net architecture. |
| Model architecture | `Unet.py` | Defines the multi-branch U-Net with attention used for both training and inference. |
| Dataset loaders | `TaxaDataset.py`, `TaxaDataset_smoothviz.py`, `TaxaDataset_smoothviz_prediction.py` | PyTorch datasets for random crop sampling, smooth-visualization monitoring, and large-scale prediction tiling. |
| Co-occurrence & embeddings | `CooccurrenceHelper.py`, `EmbeddingHelpers.py` | Helpers for building GBIF co-occurrence matrices and training Skip-gram embeddings. |
| Utilities | `Utils.py`, `RasterHelper.py`, `Utils_R.R` | Shared raster/HDF5 utilities and plotting/evaluation helpers. |
| Scripts & notebooks | `01_prepare_data.ipynb`, `02_train_deepsdm.py`, `03_make_prediction.ipynb`, `Fig*.ipynb` | Complete workflow from data harmonization through training, inference, and figure reproduction. |
| R workflow | `run_maxent_and_evaluate_models.R`, `evaluate_models_constantthreshold.R` | MaxEnt baseline training and threshold evaluation routines. |

## Execution Flow

1. **Stage assets**
   - Download and extract every archive from Google Drive so `workspace/`, `medium/`, `mlruns/`, `predicts/`, `predicts_maxent/`, `plots/`, and `raw/` sit inside the `DeepSDM/` repository folder.
   - Verify that `DeepSDM_conf.yaml` paths (e.g., `workspace_dir`, `raw_dir`) point to these sibling directories.

2. **Prepare data — `01_prepare_data.ipynb`**
   - Run the notebook cell by cell. It ingests environmental rasters from `raw/`, harmonizes them with `RasterHelper`, filters GBIF records, constructs species co-occurrence tables with `CooccurrenceHelper`, and exports embeddings via `EmbeddingHelpers`.
   - The notebook populates `workspace/` and `medium/` with cached tensors, metadata JSON files, and preprocessed rasters referenced throughout the pipeline.

3. **Train DeepSDM — `02_train_deepsdm.py`**
   - Ensure `DeepSDM_conf.yaml` reflects the desired species list, temporal window, cache directories, and trainer settings.
   - Launch training from the repository root:
     ```bash
     python 02_train_deepsdm.py
     ```
   - The script constructs `LitDeepSDMData` and `LitUNetSDM`, then logs metrics, artifacts, and checkpoints to `mlruns/` via MLflow.

4. **Track runs and checkpoints**
   - Optionally start the MLflow UI (`mlflow ui`) in the directory that contains `mlruns/` to inspect losses, F1 scores, and saved checkpoints during or after training.

5. **Generate predictions — `03_make_prediction.ipynb`**
   - Open the notebook, set `experiment_id` and `run_id` to an existing MLflow run, and load the associated `DeepSDM_conf.yaml` from `mlruns/<experiment>/<run>/artifacts/conf/`.
   - The notebook instantiates `LitDeepSDMData_prediction` and `LitUNetSDM` (from `LitUNetSDM_prediction.py`) to stream tiles across species and months, writing HDF5/PNG outputs to `predicts/` and optionally attention heatmaps.

6. **Train MaxEnt baselines — `run_maxent_and_evaluate_models.R`**
   - Edit the hard-coded `run_id` and `exp_id` near the top of the script so they match the DeepSDM run you want to compare.
   - Execute subsets of species with:
     ```bash
     Rscript run_maxent_and_evaluate_models.R <start_index>
     ```
     Each invocation processes five species at a time, reading DeepSDM predictions from `predicts/` and writing MaxEnt outputs to `predicts_maxent/`.

7. **Evaluate fixed thresholds — `evaluate_models_constantthreshold.R`**
   - Use the same `run_id` and `exp_id` configuration, then run:
     ```bash
     Rscript evaluate_models_constantthreshold.R <start_index>
     ```
     The script reloads both DeepSDM and MaxEnt predictions to recompute binary rasters and metrics under constant thresholds.

8. **Batch execution (optional)**
   - `run_maxent_and_evaluate_models_batch.sh` and `evaluate_models_constantthreshold_batch.sh` launch multiple `Rscript` jobs in parallel. Update the hard-coded index list or append a final `wait` if you rely on these helpers in a multi-core environment.

9. **Reproduce figures — `Fig*.ipynb`**
   - Execute the notebooks `Fig2_embedding.ipynb` through `Fig6_cph.ipynb`, which reuse assets from `plots/`, `mlruns/`, and `predicts/` to regenerate the manuscript and supplementary visualizations.

## Script and Module Reference

### Python
- **`02_train_deepsdm.py`** – Standalone entry point for training DeepSDM; reads `DeepSDM_conf.yaml`, initializes the Lightning data module and model, and starts MLflow logging.
- **`03_make_prediction.ipynb`** – Notebook-driven inference workflow that imports `LitDeepSDMData_prediction` and `LitUNetSDM` to restore checkpoints and tile predictions.
- **`01_prepare_data.ipynb`** – Notebook that harmonizes raw data, builds co-occurrence matrices, trains embeddings, and caches tensors for training.
- **`LitUNetSDM.py`** – Lightning module wrapping the U-Net architecture, including training/validation steps, logging, and metric computation. Imported by the training script and diagnostic notebooks.
- **`LitUNetSDM_prediction.py`** – Prediction-focused Lightning module with helper utilities for writing attention maps; imported exclusively by the inference notebook.
- **`LitDeepSDMData.py`** – Lightning data module responsible for caching tensors, constructing dataloaders, and exposing metadata for training and validation.
- **`LitDeepSDMData_prediction.py`** – Data module optimized for streaming tiles during inference; the prediction notebook instantiates it directly.
- **`Unet.py`** – Core neural network that fuses environmental rasters with species embeddings using attention at skip connections.
- **`TaxaDataset.py`** – Base PyTorch dataset that samples random crops aligned with survey-effort partitions; constructed within `LitDeepSDMData`.
- **`TaxaDataset_smoothviz.py`** / **`TaxaDataset_smoothviz_prediction.py`** – Dataset variants used for smooth visualization during training and inference monitoring.
- **`CooccurrenceHelper.py`** – Functions for filtering GBIF records, building co-occurrence matrices, and preparing species metadata used in embedding training.
- **`EmbeddingHelpers.py`** – Implements Skip-gram embedding training and serialization for use within `Unet.py`.
- **`RasterHelper.py`** – Utilities for resampling, mosaicking, and normalizing environmental rasters consumed by the preparation notebook.
- **`Utils.py`** – General-purpose tools for inspecting HDF5 outputs, plotting diagnostics, and validating prediction artifacts.

### R
- **`run_maxent_and_evaluate_models.R`** – Trains MaxEnt baselines and computes evaluation metrics; executed with `Rscript` after setting `run_id`/`exp_id`.
- **`evaluate_models_constantthreshold.R`** – Applies constant-threshold evaluation to DeepSDM and MaxEnt predictions for matched species batches; also run via `Rscript` with the appropriate indices.
- **`Utils_R.R`** – Helper functions sourced by the R scripts for I/O, plotting, and evaluation; not meant to be executed directly.

### Shell
- **`run_maxent_and_evaluate_models_batch.sh`** – Convenience script that spawns multiple instances of `run_maxent_and_evaluate_models.R` for different species index ranges.
- **`evaluate_models_constantthreshold_batch.sh`** – Parallel launcher for `evaluate_models_constantthreshold.R`. Customize the index list and add synchronization as needed.

## Primary Outputs
- `predicts/<RUN>/h5/<SPECIES>/<SPECIES>.h5` – Continuous suitability mosaics for each species and month with geospatial metadata.
- `predicts/<RUN>/png/<SPECIES>/*.png` – Visualization-ready maps combining predicted suitability with occurrence overlays.
- `predicts/<RUN>/attention/` – Optional attention heatmaps exported during inference when enabled.
- `mlruns/<EXPERIMENT>/<RUN>/` – MLflow run directories containing checkpoints, metrics, tensorboard logs, and configuration snapshots.
- `predicts_maxent/<RUN>/` – MaxEnt raster outputs, binary masks, and evaluation summaries aligned with DeepSDM runs.

## Additional Resources
- Review the source files listed above for implementation details and inline comments.
- Use the `Fig*.ipynb` notebooks to explore embeddings, attention patterns, niche-space clustering, and survival analyses featured in the manuscript.
