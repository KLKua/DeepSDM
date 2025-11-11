<<<<<<< ours
# DeepSDM 程式與模型說明

本專案提供論文中 Deep Species Distribution Model (DeepSDM) 的完整實作，涵蓋資料前處理、物種共現嵌入訓練、以多尺度 U-Net 為核心的時空分佈模型、以及推論與評估流程。程式碼以 Python 實作深度學習與資料處理，並搭配 R 腳本產出 MaxEnt 基準模型與統計分析圖表。

## 專案結構總覽

| 類別 | 主要檔案 | 功能摘要 |
| ---- | -------- | -------- |
| 設定檔 | `DeepSDM_conf.yaml` | 集中管理地理範圍、環境因子、物種名錄、訓練超參數與工作流程設定。 |
| PyTorch Lightning 模組 | `LitDeepSDMData.py`, `LitUNetSDM.py`, `LitUNetSDM_prediction.py` | 負責資料載入與快取、模型訓練與評估、批次推論與結果輸出。 |
| 模型結構 | `Unet.py` | 多分支多尺度 U-Net，結合物種嵌入注意力與環境棧影像輸入。 |
| 資料集 | `TaxaDataset.py`, `TaxaDataset_smoothviz.py`, `TaxaDataset_smoothviz_prediction.py` | 依地理切割區塊產生訓練/驗證/平滑視覺化所需的取樣資料。 |
| 共現與嵌入 | `CooccurrenceHelper.py`, `EmbeddingHelpers.py` | 建立共現單元、計算物種共現次數，並以 Skip-gram 訓練 64 維嵌入向量。 |
| 工具 | `Utils.py`, `RasterHelper.py`, `CooccurrenceHelper.py` | 提供 HDF5/Raster 處理、座標轉換、統計指標等通用函式。 |
| 腳本/筆記本 | `01_prepare_data.ipynb`, `02_train_deepsdm.py`, `03_make_prediction.ipynb` 與 `Fig*.ipynb` | 依序進行資料準備、模型訓練、推論，以及論文圖表重製。 |
| R 工作流程 | `run_maxent_and_evaluate_models.R`, `evaluate_models_constantthreshold.R`, `Utils_R.R` | 訓練傳統 MaxEnt、套用閾值並評估，輔助比較深度模型。 |

## 環境安裝
=======
# DeepSDM Code and Model Overview

This repository contains the full implementation of the Deep Species Distribution Model (DeepSDM) described in the accompanying paper. It covers end-to-end data preparation, species co-occurrence embedding training, the multi-scale U-Net model, as well as inference and evaluation workflows. Python code powers the deep learning and data processing pipelines, while the R scripts reproduce MaxEnt baselines and statistical visualizations.
>>>>>>> theirs

### Python
1. 依需求建立 Conda 或 Python 虛擬環境，建議搭配 CUDA GPU。
2. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```
   或使用 `python_env.yaml` 匯入 Conda 環境。

<<<<<<< ours
### R
1. 建立獨立的 R 環境（建議使用 renv）。
2. 依 `requirements_r.txt` 安裝套件，並參考 `setup_r_environment.md` 完成 MaxEnt 相關依賴。

## 關鍵設定與資料結構

- **工作目錄 (`./workspace/`)**：需包含研究區域遮罩 `extent_binary.tif`、地圖切割 `partition.txt`/`extent_partition.txt`、以及 `meta_json_files` 指向的環境統計與物種資訊 JSON。【F:TaxaDataset.py†L15-L45】【F:LitDeepSDMData.py†L24-L58】
- **設定檔 (`DeepSDM_conf.yaml`)**：定義訓練日期區間、物種清單、環境變數、資料快取路徑、訓練超參數與硬體配置；亦描述環境原始資料來源與共現統計參數。【F:DeepSDM_conf.yaml†L1-L210】【F:DeepSDM_conf.yaml†L330-L417】
- **快取檔 (`./tmp/*.pth`)**：資料模組啟動時會將環境棧、標籤、k2 權重與嵌入依訓練/驗證/視覺化階段快取，避免重複 I/O。【F:LitDeepSDMData.py†L79-L148】

## 典型使用流程

1. **資料準備 (`01_prepare_data.ipynb`)**
   - 下載並裁切 CHELSA、MODIS、ESA CCI 等環境變數，統計每月平均與標準差，寫入 `env_information.json`。
   - 透過 `CooccurrenceHelper` 濾除原始 GBIF 紀錄，依空間/時間網格累計共現次數；結果儲存於 `workspace/species_data`。【F:CooccurrenceHelper.py†L10-L131】【F:CooccurrenceHelper.py†L133-L211】
   - 使用 `EmbeddingHelpers` 讀取共現表，採 Skip-gram + 負採樣訓練物種嵌入，並輸出最佳權重與指標檔。【F:EmbeddingHelpers.py†L13-L126】【F:EmbeddingHelpers.py†L150-L226】
   - 產生物種出現 HDF5 格網與 `meta_json_files` 描述各項資料。

2. **模型訓練 (`02_train_deepsdm.py`)**
   - 透過 `LitDeepSDMData` 模組載入資料並於多 GPU 上建立 Lightning DataLoader；批次大小、抽樣高度寬度等參數由設定檔控制。【F:LitDeepSDMData.py†L15-L138】【F:DeepSDM_conf.yaml†L9-L76】
   - `LitUNetSDM` 以多分支 U-Net 結合物種嵌入注意力，採努力權重的 BCE Loss，並於每輪驗證後計算最佳閾值的 F1 分數做為監控指標，支援 MLflow 記錄。【F:Unet.py†L1-L148】【F:LitUNetSDM.py†L21-L130】【F:LitUNetSDM.py†L164-L256】
   - Lightning `Trainer` 使用 DDP 多 GPU、模型檢查點與提前停止機制，主要監控 `f1_train`。【F:02_train_deepsdm.py†L3-L68】

3. **推論 (`03_make_prediction.ipynb` / `LitUNetSDM_prediction.py`)**
   - 載入訓練好的檢查點，建立預測 DataLoader 並巡覽所有物種-日期影格。
   - 模型輸出會重建完整地理網格，計算切片平均後寫入 HDF5 與 PNG；若開啟 `predict_attention` 亦會輸出各環境變數注意力圖。【F:LitUNetSDM_prediction.py†L15-L160】【F:LitUNetSDM_prediction.py†L162-L256】

4. **評估與比較**
   - `run_maxent_and_evaluate_models.R` / `evaluate_models_constantthreshold.R` 讀取相同的輸入資料，訓練 MaxEnt 並產出多種閾值的二元棲地圖與指標，用於與 DeepSDM 比較。
   - `Fig*.ipynb` 筆記本重製論文中嵌入、注意力、棲地位、聚類與生存分析等圖。

## 模型設計重點

- **注意力式環境融合**：物種嵌入向量經全連接層產生查詢向量，與環境影像卷積後形成鍵值，透過 Softmax 注意力對環境棧加權，再與嵌入特徵於各層跳接融合。【F:Unet.py†L12-L118】
- **努力權重損失**：利用 `k2` 層記錄調查努力，將 BCE 損失分為出現點、調查但未出現、無調查三類，並以設定檔的 `k2`、`k3` 權重組合總損失。【F:LitUNetSDM.py†L49-L121】
- **自動閾值搜尋**：每輪驗證計算多個閾值的 F1，選取最佳閾值並記錄加權平均 F1，提供模型檢查點依據。【F:LitUNetSDM.py†L210-L262】
- **區塊隨機取樣**：`TaxaDataset` 依 `partition.txt` 切割地圖，僅對具有調查努力的區塊建立可用組合，並在每次 `__getitem__` 隨機裁切子圖增加資料多樣性。【F:TaxaDataset.py†L15-L120】【F:TaxaDataset.py†L134-L185】

## 主要輸出

- `predicts/<RUN>/h5/<SPECIES>/<SPECIES>.h5`：連續棲地適宜度，含 CRS、變換矩陣與 NoData 符號屬性。【F:LitUNetSDM_prediction.py†L170-L215】
- `predicts/<RUN>/png/<SPECIES>/*.png`：每月棲地適宜度與觀測點疊圖。【F:LitUNetSDM_prediction.py†L181-L205】
- `predicts/<RUN>/attention/`：若啟用注意力輸出，各環境變數的注意力熱圖以 HDF5 儲存。【F:LitUNetSDM_prediction.py†L217-L256】
- `mlruns/`：PyTorch Lightning 透過 MLflow 自動紀錄訓練指標與超參數。【F:02_train_deepsdm.py†L44-L68】

## 常見調整項

- 更新訓練物種或環境變數時，需同步修改 `DeepSDM_conf.yaml` 的 `species_list_*` 與 `env_list`，並重新執行資料快取與嵌入訓練流程。【F:DeepSDM_conf.yaml†L41-L209】
- 若 GPU 數量不同，可在設定檔 `trainer_conf.devices` 或 `02_train_deepsdm.py` 中調整 Lightning Trainer 設定。【F:DeepSDM_conf.yaml†L307-L317】【F:02_train_deepsdm.py†L48-L68】
- 預測時可透過 `predict_attention=True` 產生注意力圖，或修改 `DeepSDM_conf.training_conf.num_predict_steps` 控制批次大小與速度。【F:LitUNetSDM_prediction.py†L18-L112】【F:DeepSDM_conf.yaml†L19-L34】

---
本 README 根據原始程式碼整理，協助快速復現論文中的 DeepSDM 模型與周邊分析。若需了解更細節的數據處理或圖像生成流程，請參閱各筆記本與模組內註解。
=======
| Category | Key Files | Summary |
| --- | --- | --- |
| Configuration | `DeepSDM_conf.yaml` | Central configuration for geographic extent, environmental predictors, species lists, training hyperparameters, and workflow switches. |
| PyTorch Lightning modules | `LitDeepSDMData.py`, `LitUNetSDM.py`, `LitUNetSDM_prediction.py` | Data loading and caching, model training and validation, and batched inference utilities. |
| Model architecture | `Unet.py` | Multi-branch multi-scale U-Net with species embedding attention and environmental raster stacks. |
| Datasets | `TaxaDataset.py`, `TaxaDataset_smoothviz.py`, `TaxaDataset_smoothviz_prediction.py` | Dataset definitions that sample spatial tiles for training/validation/visualization. |
| Co-occurrence and embeddings | `CooccurrenceHelper.py`, `EmbeddingHelpers.py` | Builds co-occurrence units, aggregates counts, and trains skip-gram embeddings for species. |
| Utilities | `Utils.py`, `RasterHelper.py`, `CooccurrenceHelper.py` | Shared helpers for HDF5/raster I/O, coordinate transforms, statistics, and logging. |
| Scripts / notebooks | `01_prepare_data.ipynb`, `02_train_deepsdm.py`, `03_make_prediction.ipynb`, `Fig*.ipynb` | Sequential notebooks and scripts for preparation, training, inference, and figure reproduction. |
| R workflow | `run_maxent_and_evaluate_models.R`, `evaluate_models_constantthreshold.R`, `Utils_R.R` | Builds MaxEnt baselines, applies thresholds, and evaluates metrics for comparisons. |

## Environment Setup

### Python
1. Create a Conda or virtualenv environment (CUDA-enabled GPU recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   or import the Conda environment specified in `python_env.yaml`.

### R
1. Set up a dedicated R environment (e.g., via renv).
2. Install packages listed in `requirements_r.txt` and follow `setup_r_environment.md` for MaxEnt-related dependencies.

## Key Configuration and Data Layout

- **Workspace (`./workspace/`)** – Must contain the study-area mask `extent_binary.tif`, tiling definitions `partition.txt`/`extent_partition.txt`, and the `meta_json_files` referenced in the configuration. These JSON files describe environmental statistics and species metadata required by the datasets and datamodules.【F:TaxaDataset.py†L15-L45】【F:LitDeepSDMData.py†L24-L58】
- **Configuration file (`DeepSDM_conf.yaml`)** – Defines training time windows, species lists, environmental predictors, cache directories, training hyperparameters, and hardware settings, together with the provenance of the environmental layers and co-occurrence aggregation parameters.【F:DeepSDM_conf.yaml†L1-L210】【F:DeepSDM_conf.yaml†L330-L417】
- **Cache files (`./tmp/*.pth`)** – Generated by the data modules to persist environmental stacks, labels, effort weights, and embeddings for train/validation/visualization splits, minimizing repeated disk I/O.【F:LitDeepSDMData.py†L79-L148】

## Typical Workflow

1. **Data preparation (`01_prepare_data.ipynb`)**
   - Download and clip environmental predictors (CHELSA, MODIS, ESA CCI), compute monthly means/standard deviations, and write them to `env_information.json`.
   - Use `CooccurrenceHelper` to filter GBIF records, aggregate co-occurrence counts over the spatiotemporal grid, and store results under `workspace/species_data`.【F:CooccurrenceHelper.py†L10-L211】
   - Train species embeddings with `EmbeddingHelpers` using skip-gram with negative sampling, saving the learned weights and evaluation metadata.【F:EmbeddingHelpers.py†L13-L226】
   - Produce the HDF5 occurrence cubes and populate `meta_json_files` with references to all derived assets.

2. **Model training (`02_train_deepsdm.py`)**
   - `LitDeepSDMData` instantiates Lightning DataLoaders that tile the raster stacks across GPUs according to batch/patch sizes defined in the configuration.【F:LitDeepSDMData.py†L15-L138】【F:DeepSDM_conf.yaml†L9-L76】
   - `LitUNetSDM` combines multi-branch U-Net encoders with species embedding attention, optimizes an effort-weighted BCE loss, and logs validation metrics including thresholded F1 scores via MLflow.【F:Unet.py†L1-L148】【F:LitUNetSDM.py†L21-L256】
   - The Lightning `Trainer` leverages DDP multi-GPU execution, checkpointing, and early stopping while monitoring `f1_train`.【F:02_train_deepsdm.py†L3-L68】

3. **Inference (`03_make_prediction.ipynb` / `LitUNetSDM_prediction.py`)**
   - Load saved checkpoints, build prediction DataLoaders, and iterate over species-month tensors.
   - Reconstruct full-extent raster predictions, aggregate tiles, and export HDF5/PNG outputs. Optional attention maps can be persisted when `predict_attention` is enabled.【F:LitUNetSDM_prediction.py†L15-L256】

4. **Evaluation and comparison**
   - `run_maxent_and_evaluate_models.R` and `evaluate_models_constantthreshold.R` reuse the same inputs to train MaxEnt models and compute threshold-dependent binary maps and metrics for benchmarking.
   - `Fig*.ipynb` notebooks recreate embedding, attention, niche space, clustering, and survival analysis figures from the paper.

## Model Highlights

- **Attention-based environmental fusion** – Species embeddings are projected into queries that attend over convolutional keys/values derived from environmental stacks, allowing per-species weighting across skip connections.【F:Unet.py†L12-L118】
- **Effort-aware loss** – The `k2` effort layer separates observed presences, surveyed absences, and unsurveyed areas, combining BCE terms weighted by configuration-defined `k2`/`k3` factors.【F:LitUNetSDM.py†L49-L121】
- **Automatic threshold search** – Validation computes F1 across candidate thresholds, stores the best value, and uses it for checkpoint selection.【F:LitUNetSDM.py†L210-L262】
- **Partition-based sampling** – `TaxaDataset` restricts patches to partitions with survey effort and performs random crops per `__getitem__` call to increase diversity.【F:TaxaDataset.py†L15-L185】

## Main Outputs

- `predicts/<RUN>/h5/<SPECIES>/<SPECIES>.h5`: Continuous habitat suitability rasters with CRS, transform, and NoData attributes.【F:LitUNetSDM_prediction.py†L170-L215】
- `predicts/<RUN>/png/<SPECIES>/*.png`: Monthly suitability maps with observation overlays.【F:LitUNetSDM_prediction.py†L181-L205】
- `predicts/<RUN>/attention/`: Optional attention heatmaps for each environmental predictor when enabled.【F:LitUNetSDM_prediction.py†L217-L256】
- `mlruns/`: MLflow experiment logs created by PyTorch Lightning during training.【F:02_train_deepsdm.py†L44-L68】

## Common Adjustments

- Update `species_list_*` and `env_list` in `DeepSDM_conf.yaml` when adding species or predictors, then regenerate caches and retrain embeddings.【F:DeepSDM_conf.yaml†L41-L209】
- Adjust `trainer_conf.devices` or override Trainer settings in `02_train_deepsdm.py` to match available GPUs.【F:DeepSDM_conf.yaml†L307-L317】【F:02_train_deepsdm.py†L48-L68】
- Enable `predict_attention=True` to export attention maps, or tweak `DeepSDM_conf.training_conf.num_predict_steps` to balance throughput and memory during inference.【F:LitUNetSDM_prediction.py†L18-L112】【F:DeepSDM_conf.yaml†L19-L34】

---
This README compiles the essential details from the source code to help reproduce the DeepSDM experiments and extend the model to new regions or species. Refer to the notebooks and inline comments for deeper explanations of data processing and visualization steps.
>>>>>>> theirs
