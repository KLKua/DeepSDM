# DeepSDM: Deep Species Distribution Modeling Framework

DeepSDM is a deep learning framework for modeling species distributions using environmental data and species co-occurrence patterns. This framework leverages attention mechanisms to capture important environmental factors and produces high-quality species distribution predictions.

## Data and Results

All data and results for this project are available at:
https://drive.google.com/drive/folders/1zzJg_q1gTyvoprR7r4iX69xrOYRsJlGR?usp=drive_link

## Overview

DeepSDM uses a U-Net architecture with attention mechanisms to predict species distributions based on environmental variables and species embeddings derived from co-occurrence data. The framework consists of several components for data preparation, model training, prediction, and evaluation.

## System Requirements

### Software Requirements
- Python 3.8+
- PyTorch 1.13.1
- PyTorch Lightning 2.0.6
- GDAL
- Various Python packages (rasterio, umap-learn, mlflow, etc.)

### Hardware Requirements
- CUDA-compatible GPU (recommended)
- Sufficient RAM for processing large environmental datasets

## Installation

```bash
# System dependencies
sudo apt install build-essential
sudo apt update
sudo apt install libpq-dev
sudo apt install software-properties-common
sudo apt-add-repository ppa:ubuntugis/ppa
sudo apt install gdal-bin
sudo apt install libgdal-dev
sudo apt install libgl1-mesa-glx

# Check GDAL version and install matching Python package
gdalinfo --version
pip install gdal==<version>

# Install other required packages
pip install rasterio umap-learn mlflow
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install pytorch-lightning==2.0.6 torchmetrics==0.11.0
```

## Workflow

The DeepSDM workflow consists of the following steps:

### 0. Download Environmental Data (Download_env.ipynb)

This Jupyter notebook provides instructions and code for downloading the necessary environmental data:

- **CHELSA Dataset**: Downloads climate variables (clt, hurs, pr, rsds, sfcWind, tas) for each month from 2000 to 2019
- **Land Cover Dataset**: Instructions for downloading ESA Land Cover data (2000-2020)
- **EVI Dataset**: Instructions for obtaining Enhanced Vegetation Index data from NASA's AppEEARS
- **Elevation Dataset**: Link to download WorldClim Elevation Data

This step is optional if you already have the environmental data available in the Google Drive link.

### 1. Data Preparation (01_prepare_data.ipynb)

This Jupyter notebook handles all data processing steps before training:

- **Configuration Loading**: Loads parameters from `DeepSDM_conf.yaml` 
- **Spatial Configuration**: Creates extent maps and train/validation splits
- **Environmental Data Processing**: 
  - Processes raw environmental rasters to align with the defined extent
  - Normalizes values and handles missing data
  - For land cover data, performs PCA to reduce dimensionality
  - Averages data across time spans
- **Species Occurrence Processing**:
  - Filters raw GBIF occurrence data
  - Creates aligned species presence rasters
  - Generates effort-weight (k) rasters for loss calculation
- **Species Co-occurrence Embeddings**:
  - Identifies co-occurring species
  - Trains embeddings to capture ecological relationships
  - Creates a vector representation for each species

### 2. Model Training (02_train_deepsdm.py)

This script orchestrates the training process:

- **Configuration and Initialization**:
  - Initializes the data module and model
- **Checkpoint and Early Stopping Setup**:
  - Configures model checkpointing to save the best models based on F1 score
  - Sets up early stopping to prevent overfitting
- **Trainer Configuration**:
  - Initializes PyTorch Lightning Trainer with specified devices
  - Configures distributed data parallel training (DDP) for multi-GPU usage
  - Sets up MLflow logger for experiment tracking
- **Model Training**:
  - Performs model training with effort-weighted loss
  - Conducts periodic validation
  - Logs progress to MLflow

### 3. Prediction (03_make_prediction.ipynb)

This notebook handles generating predictions with the trained model:

- **Model Loading**:
  - Loads saved model checkpoints from MLflow
  - Can load and average the top-k best models
- **Single GPU Prediction**:
  - Simple prediction loop for one GPU
- **Multi-GPU Prediction**:
  - Distributes prediction tasks across multiple GPUs
  - Splits species and dates into separate batches
- **Output Generation**:
  - Saves predictions as GeoTIFF files
  - Creates visualization images (PNG)
  - Optionally generates attention maps

### 4. Evaluation (run_maxent_and_evaluate_models.R, evaluate_models_constantthreshold.R)

Two R scripts handle model evaluation and comparison:

- **run_maxent_and_evaluate_models.R**:
  - Runs MaxEnt models on the same data for comparison
  - Calculates performance metrics (AUC, TSS, Kappa, F1)
  - Creates summary statistics and comparison tables
- **evaluate_models_constantthreshold.R**:
  - Evaluates models using constant thresholds across dates
  - Generates binary prediction maps
  - Calculates threshold-dependent metrics

**Important**: To execute these evaluation scripts correctly, you must:
1. Copy the contents of `run_maxent_and_evaluate_models_batch.sh` to your terminal and execute first
2. After that completes, copy the contents of `evaluate_models_constantthreshold_batch.sh` to your terminal and execute
3. Make sure to maintain this exact execution order as the second script depends on outputs from the first script

## Code Structure

### Core Files and Their Functions

- **01_prepare_data.ipynb**: Data preparation notebook
- **02_train_deepsdm.py**: Model training script
- **03_make_prediction.ipynb**: Prediction notebook

### Model Architecture Files

- **Unet.py**: Implements the core U-Net architecture with attention mechanisms
- **LitUNetSDM.py**: PyTorch Lightning module that wraps the U-Net model
- **LitUNetSDM_prediction.py**: Modified model module for prediction

### Data Handling Files

- **LitDeepSDMData.py**: Data module for PyTorch Lightning
- **LitDeepSDMData_prediction.py**: Modified data module for prediction
- **TaxaDataset.py**: Dataset class for training data
- **TaxaDataset_smoothviz.py**: Dataset class for visualization during training
- **TaxaDataset_smoothviz_prediction.py**: Dataset class for prediction

### Utility Files

- **RasterHelper.py**: Handles raster processing and manipulation
- **CooccurrenceHelper.py**: Calculates species co-occurrences
- **EmbeddingHelpers.py**: Trains species embeddings
- **Utils.py**: Utility functions for Python
- **Utils_R.R**: Utility functions for R

### Configuration File

- **DeepSDM_conf.yaml**: Contains all parameters for the framework:
  - File paths for data
  - Training configurations
  - Spatial and temporal settings
  - Model parameters
  - Lists of species and dates

## Key Components

### Spatial Configuration

The spatial unit for training is defined by:
- Geographic extent (x_start, y_start, x_end, y_end)
- Grid size and resolution
- Train/validation splits

### Temporal Configuration

The temporal unit is defined by:
- Date range (date_start, date_end)
- Time span parameters (month_span, month_step)
- Co-occurrence time limit

### Environmental Data

DeepSDM supports various environmental factors:
- Cloud area fraction (clt)
- Relative humidity (hurs)
- Precipitation (pr)
- Shortwave radiation (rsds)
- Wind speed (sfcWind)
- Temperature (tas)
- Enhanced Vegetation Index (EVI)
- Land cover principal components

### Model Architecture

The model uses a U-Net architecture with:
- Attention mechanisms to focus on relevant environmental factors
- Species embeddings as input
- Skip connections to preserve spatial information
- Multiple convolutional layers for complex patterns

### Loss Function

The model uses a weighted binary cross-entropy loss with three components:
1. Loss for presence points
2. Effort-weighted loss for surveyed absence points
3. Background loss for unsurveyed pixels

## Usage Examples

### Complete Workflow

```bash
# 1. Prepare data
jupyter notebook 01_prepare_data.ipynb
# Execute all cells to process data

# 2. Train model
python 02_train_deepsdm.py
# Monitor training progress with MLflow
mlflow ui

# 3. Make predictions
jupyter notebook 03_make_prediction.ipynb
# Execute cells to generate predictions

# 4. Evaluate results
# Copy the contents of run_maxent_and_evaluate_models_batch.sh to terminal and execute
# After it completes, copy the contents of evaluate_models_constantthreshold_batch.sh to terminal
# Note: These must be executed in this exact order!
```

### Custom Species and Dates

To predict custom species and dates, modify the YAML configuration or edit the prediction notebook:

```yaml
training_conf:
  species_list_predict: 
    - Carpodacus_formosanus
    - Parus_monticolus
  date_list_predict:
    - '2018-01-01'
    - '2018-07-01'
    - '2018-10-01'
```

## Directory Structure

```
DeepSDM/
├── 01_prepare_data.ipynb        # Data preparation notebook
├── 02_train_deepsdm.py          # Model training script
├── 03_make_prediction.ipynb     # Prediction notebook
├── CooccurrenceHelper.py        # Helper for co-occurrences
├── DeepSDM_conf.yaml            # Configuration file
├── Download_env.ipynb           # Instructions for downloading environmental data 
├── EmbeddingHelpers.py          # Helper for embeddings
├── LitDeepSDMData.py            # Data module
├── LitDeepSDMData_prediction.py # Data module for prediction
├── LitUNetSDM.py                # Model module
├── LitUNetSDM_prediction.py     # Model module for prediction
├── RasterHelper.py              # Helper for raster data
├── TaxaDataset.py               # Dataset class
├── TaxaDataset_smoothviz.py     # Dataset for visualization
├── TaxaDataset_smoothviz_prediction.py # Dataset for prediction
├── Unet.py                      # U-Net model
├── Utils.py                     # Utilities
├── Utils_R.R                    # R utilities
├── dwca-trait_454-v1.68/        # Trait dataset files
├── evaluate_models_constantthreshold.R # Evaluation script
├── evaluate_models_constantthreshold_batch.sh # Batch script for evaluation
├── mlruns/                      # MLflow tracking
├── plots/                       # Result analysis files and paper figures
├── predicts/                    # DeepSDM model predictions
├── predicts_maxent/             # MaxEnt results and model performance
├── raw/                         # Raw input data
├── run_maxent_and_evaluate_models.R   # MaxEnt comparison
├── run_maxent_and_evaluate_models_batch.sh # Batch script for MaxEnt
└── workspace/                   # Processed data
```

## Data Formats and Storage

### Input Data
- Environmental rasters: GeoTIFF format
- Species occurrence: CSV from GBIF
- Land cover: NetCDF files

### Intermediate Data
- Processed environmental layers: Aligned GeoTIFF files
- Species presence rasters: Binary GeoTIFF files
- Effort (k) rasters: GeoTIFF files
- Species embeddings: JSON vectors

### Output Data
- Model predictions: GeoTIFF files (0-1 values)
- Binary predictions: Thresholded GeoTIFF files
- Attention maps: GeoTIFF files showing variable importance
- Performance metrics: CSV files