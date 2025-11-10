# Species‑Embedding & Attention Multiscale U‑Net for Species Distribution Modeling (SEAM-SDM)

## Project Overview

This project implements a deep learning-based Species Distribution Model (SDM) for predicting spatiotemporal distributions of bird species in Taiwan. The model combines Convolutional Neural Network (CNN) architecture with species co-occurrence relationships to predict habitat suitability across different temporal scales.

### Key Features

- **Deep Learning Architecture**: Modified U-Net architecture integrating environmental variables and species embeddings
- **Species Co-occurrence Learning**: Word2Vec-like approach for learning inter-species relationships
- **Spatiotemporal Prediction**: Monthly temporal resolution for species distribution forecasting
- **Model Comparison**: Systematic comparison with traditional MaxEnt models
- **Niche Space Analysis**: Species niche analysis in PCA-reduced environmental space

## Project Structure

```
.
├── 01_prepare_data.ipynb              # Data preparation workflow
├── 02_train_deepsdm.py                # Main model training script
├── 03_make_prediction.ipynb           # Model prediction workflow
├── run_maxent_and_evaluate_models.R   # MaxEnt model training & evaluation
├── evaluate_models_constantthreshold.R # Fixed threshold model evaluation
├── DeepSDM_conf.yaml                  # Configuration file
│
├── Fig2_embedding.ipynb               # Species embedding visualization
├── Fig3_attention.ipynb               # Attention mechanism analysis
├── Fig4_nichespace.ipynb              # Niche space analysis
├── Fig5_nichespace_clustering.ipynb   # Niche clustering analysis
├── Fig6_cph.ipynb                     # Cox proportional hazards analysis
│
├── Utils.py                           # Python utility functions
├── Utils_R.R                          # R utility functions
├── LitDeepSDMData.py                  # Data module
├── LitDeepSDMData_prediction.py       # Prediction data module
├── LitUNetSDM.py                      # Model training module
├── LitUNetSDM_prediction.py           # Model prediction module
├── Unet.py                            # U-Net network architecture
├── TaxaDataset.py                     # Dataset class
├── TaxaDataset_smoothviz.py           # Smooth visualization dataset
├── TaxaDataset_smoothviz_prediction.py # Prediction visualization dataset
├── EmbeddingHelpers.py                # Species embedding helpers
├── RasterHelper.py                    # Raster data processing
├── CooccurrenceHelper.py              # Co-occurrence computation
│
├── requirements.txt                   # Python dependencies
├── requirements_r.txt                 # R dependencies
├── python_env.yaml                    # Conda environment config
└── setup_r_environment.md             # R environment setup guide
```

## Methodology

### 1. Data Preparation

#### Environmental Variables
- **Climate Data**: CHELSA v2.1 (cloud cover, humidity, precipitation, radiation, wind speed, temperature, etc.)
- **Vegetation Index**: EVI (Enhanced Vegetation Index)
- **Land Cover**: ESA CCI Land Cover (PCA-reduced)
- **Temporal Coverage**: 2001-2018, monthly resolution
- **Spatial Resolution**: ~1km × 1km

#### Species Occurrence Data
- 125 Taiwan resident bird species occurrence records
- Processed into binary rasters (presence/absence)
- Spatially-split training/validation sets

### 2. Model Architecture

#### DeepSDM Network Structure

```
Inputs:
├── Environmental Variables (11-D): [clt, hurs, pr, rsds, sfcWind, tas, EVI, landcover_PC01-04]
└── Species Embedding Vector (64-D): Learned from co-occurrence matrix

Network Architecture:
├── Species Embedding Branch:
│   └── 4-layer convolutional downsampling (16→32→64→128 channels)
│
├── Main U-Net Branch:
│   ├── Encoder (Downsampling):
│   │   ├── Conv Block 1: 64 channels + species feature fusion
│   │   ├── Conv Block 2: 128 channels + species feature fusion
│   │   ├── Conv Block 3: 256 channels + species feature fusion
│   │   └── Conv Block 4: 512 channels + species feature fusion
│   │
│   └── Decoder (Upsampling):
│       ├── UpConv + Skip Connection: 256 channels
│       ├── UpConv + Skip Connection: 128 channels
│       ├── UpConv + Skip Connection: 64 channels
│       └── UpConv + Skip Connection: 32 channels
│
└── Output: 1×H×W (habitat suitability prediction)

Special Design:
- Attention Mechanism: A = softmax(species_embedding)
- Feature Fusion: multiply(A, environmental_features)
- Normalization: Group Normalization (4 groups)
- Activation: LeakyReLU
```

#### Species Embedding Learning

Word2Vec-like skip-gram model for learning species co-occurrence relationships:

```python
Loss = -Σ log(sigmoid(u·v)) - Σ log(sigmoid(-nu·nv))
       positive pairs      negative samples
```

- Positive samples: Actually co-occurring species pairs (weighted by co-occurrence frequency)
- Negative samples: Randomly sampled species pairs
- Embedding dimension: 64

### 3. Training Strategy

```yaml
Hyperparameters:
  batch_size: 350
  epochs: 300
  learning_rate: 0.0001
  optimizer: Adam
  
Loss Functions:
  k2: BCE(prediction, target) - Binary Cross Entropy
  k2_p: Smoothness penalty (0.33)
  k3: L2 regularization (0.083)
  
Data Augmentation:
  - Random cropping: 56×56 patches
  - Temporal subsampling: Use subset of time steps per training iteration
```

### 4. Evaluation Metrics

- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve
- **TSS**: True Skill Statistic
- **Kappa**: Cohen's Kappa Coefficient
- **F1-Score**: Harmonic mean of precision and recall

### Python Environment

```bash
# Create environment using conda
conda env create -f python_env.yaml
conda activate deepsdm

# Or install using pip
pip install -r requirements.txt
```

Key Dependencies:
- PyTorch >= 1.9.0
- PyTorch Lightning >= 1.5.0
- rasterio
- h5py
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn

### R Environment

```bash
# Install R packages
Rscript -e "install.packages(c('raster', 'dismo', 'rJava', 'pROC', 'tidyverse', 'rjson', 'yaml', 'hdf5r'))"
```

See `setup_r_environment.md` for detailed setup instructions.

## Usage

### Complete Workflow

#### Step 1: Data Preparation

```bash
# Run Jupyter Notebook
jupyter notebook 01_prepare_data.ipynb
```

This step will:
- Process raw environmental data
- Perform PCA dimensionality reduction (land cover)
- Generate species occurrence rasters
- Calculate species co-occurrence matrix
- Train species embedding vectors

#### Step 2: Train DeepSDM Model

```bash
python 02_train_deepsdm.py
```

Training process:
- Model training using PyTorch Lightning
- Automatic saving of best model checkpoints
- Log training metrics to MLflow
- Generate training curves and performance metrics

#### Step 3: Generate Predictions

```bash
jupyter notebook 03_make_prediction.ipynb
```

Prediction outputs:
- Species distribution probability maps (HDF5 format)
- Attention weight maps (optional)
- PNG visualization images

#### Step 4: Train MaxEnt Baseline Model

```bash
Rscript run_maxent_and_evaluate_models.R
```

MaxEnt training:
- Train independent models for each species-time combination
- Use same presence/pseudo-absence points
- Calculate same evaluation metrics as DeepSDM

#### Step 5: Model Comparison and Analysis

```bash
Rscript evaluate_models_constantthreshold.R  # Fixed threshold evaluation

# Run figure generation notebooks
jupyter notebook Fig2_embedding.ipynb
jupyter notebook Fig3_attention.ipynb  
jupyter notebook Fig4_nichespace.ipynb
jupyter notebook Fig5_nichespace_clustering.ipynb
jupyter notebook Fig6_cph.ipynb
```

## Configuration File

### DeepSDM_conf.yaml

Main configuration items:

```yaml
# Geographic extent
geo_extent_file: ./workspace/extent_binary.tif

# Training configuration
training_conf:
  batch_size_train: 350
  epochs: 300
  learning_rate: 0.0001
  env_list: ['clt', 'hurs', 'pr', 'rsds', 'sfcWind', 'tas', 'EVI', 
             'landcover_PC01', 'landcover_PC02', 'landcover_PC03', 'landcover_PC04']
  date_list_train: ['2001-01-01', '2001-02-01', ...]  # Monthly time series
  species_list_train: ['Arborophila_crudigularis', ...]  # 125 species

# Species embedding configuration
embedding_conf:
  num_vector: 64
  epochs: 2000
  batch_size: 1000
  num_neg: 10

# Environmental data source configuration
env_source_conf:
  clt: CHELSA v2.1
  hurs: CHELSA v2.1
  # ... other environmental variables
  
# Land cover PCA configuration  
CCI_conf:
  landcover:
    PCA: 0.8  # Retain 80% variance
```

## Main Module Descriptions

### Core Training Modules

**LitUNetSDM.py**: PyTorch Lightning module
- Define training loop
- Implement custom loss functions
- Handle validation and testing
- Save model checkpoints

**Unet.py**: U-Net network definition
- Custom U-Net architecture
- Species embedding feature fusion
- Attention mechanism implementation

### Data Processing Modules

**LitDeepSDMData.py**: Data module
- Define data loaders
- Implement train/validation split
- Handle batch sampling

**TaxaDataset.py**: PyTorch dataset class
- Load environmental data from HDF5 files
- Dynamically generate subsample patches
- Handle presence/background points

**CooccurrenceHelper.py**: Co-occurrence relationships
- Calculate species co-occurrence matrix
- Handle spatiotemporal overlap
- Generate species pair datasets

**EmbeddingHelpers.py**: Species embeddings
- Skip-gram model implementation
- Negative sampling strategy
- UMAP visualization

### Utility Functions

**Utils.py**: Python general utilities
- HDF5 data I/O
- Raster data processing
- Performance metric calculation

**Utils_R.R**: R general utilities
- MaxEnt model training
- Evaluation metric calculation
- Raster operation functions

**RasterHelper.py**: Raster processing
- Geographic coordinate transformation
- Raster resampling
- Spatial indexing

## Output Results

### Prediction Results

```
predicts/[RUN_ID]/
├── h5/
│   └── all/
│       └── [SPECIES]/
│           └── [SPECIES].h5          # Continuous suitability predictions
├── png/
│   └── all/
│       └── [SPECIES]/
│           └── [SPECIES]_[DATE]_*.png  # Visualization images
└── attention/
    └── [SPECIES]/
        └── [SPECIES]_[DATE]_attention.h5  # Attention weights
```

### MaxEnt Results

```
predicts_maxent/[RUN_ID]/
├── h5/
│   ├── all/                            # Continuous predictions
│   ├── binary/                         # Binary predictions (variable threshold)
│   └── binary_constantthreshold/       # Binary predictions (fixed threshold)
├── maxent_model/                       # Model objects
├── env_contribution/                   # Environmental contributions
└── model_performance_*.csv             # Performance metrics
```

### Analysis Figures

```
plots/[RUN_ID]/
├── Fig2_embedding/                     # Species embedding UMAP plots
├── Fig3_attention/                     # Attention weight heatmaps
├── Fig4_nichespace/                    # Niche space analysis
├── Fig5_nichespace_clustering/         # Niche clustering results
└── Fig6_cph/                           # Survival analysis results
```

## Evaluation and Results

### Model Performance

Main advantages of DeepSDM compared to MaxEnt:

1. **Temporal Generalization**: Better prediction of species distributions at unseen time points
2. **Data Efficiency**: Better performance for data-sparse species through information sharing via species embeddings
3. **Spatial Consistency**: Spatially smoother and more continuous predictions
4. **Niche Accuracy**: Predictions in niche space better match actual species distributions


### Ecological Insights

1. **Environmental Gradients**: Temperature, precipitation, and elevation are primary drivers of species distributions
2. **Species Co-occurrence**: Embedding space reveals clustering patterns of functionally similar species
3. **Temporal Dynamics**: Model captures seasonal migration and temporal variation in habitat use
4. **Spatial Heterogeneity**: Attention mechanism identifies differential species responses to environmental factors

## Advanced Features

### Custom Species

To make predictions for new species:

1. Prepare species occurrence data (CSV format with longitude, latitude, and date)
2. Update species list in `DeepSDM_conf.yaml`
3. Re-run data preparation workflow to generate HDF5 files
4. For species not in original training set, use zero-shot prediction (with default embedding vector)

### Custom Environmental Variables

To add new environmental variables:

1. Prepare monthly GeoTIFF raster data
2. Add configuration in `env_source_conf` of `DeepSDM_conf.yaml`
3. Update `env_list` to include new variable name
4. Retrain model

### Transfer to Other Regions

To apply model to a new region:

1. Prepare environmental variable rasters for new region
2. Create new `extent_binary.tif` defining study area
3. Update geographic extent settings in `DeepSDM_conf.yaml`
4. Use pre-trained model for transfer learning or fully retrain


### Performance Optimization

- Use GPU acceleration for training (CUDA-enabled PyTorch)
- Enable mixed precision training (`precision=16` in Trainer)
- Use multi-GPU training (`gpus=[0,1,2,3]`)
- Increase number of data loader workers

---

**Note**: This README provides a comprehensive overview of the project. For detailed technical explanations, please refer to the docstrings in individual modules and the manuscript.