# Deforestation Spread Prediction in TI Koatinemo

Predicting the direction and intensity of deforestation spread within Terra Indigena Koatinemo (Para, Brazil) using Sentinel-2 NDMI time series and ConvLSTM2D with Monte Carlo Dropout uncertainty estimation.

## Repository Structure

```
├── data/                  # Sentinel-2 bands + TI boundary GeoJSON
│   ├── bands/             # B08, B11, SCL GeoTIFFs (not tracked, download with src/)
│   └── ti_koatinemo.geojson
├── src/                   # Data download and model training scripts
│   ├── download_bands.py  # Downloads Sentinel-2 data via STAC API
│   └── train_convlstm.py  # Standalone ConvLSTM training script
├── notebooks/             # Analysis notebooks
│   ├── thesis_pipeline.ipynb  # Main end-to-end pipeline
│   ├── explore.ipynb          # Data exploration
│   └── predict.ipynb          # Prediction experiments
├── models/
│   └── model.pt           # Trained ConvLSTM2D weights (included in repo)
├── figures/               # Generated figures (PDF)
```

## How to Replicate Results

### 1. Environment Setup

```bash
# Create a conda environment (Python 3.10+)
conda create -n deforestation python=3.12
conda activate deforestation

# Install dependencies
pip install numpy torch rasterio matplotlib scipy shapely pyproj
pip install jupyter
```

### 2. Download Satellite Data

```bash
# Downloads Sentinel-2 B08, B11, SCL bands for TI Koatinemo area
# Requires internet connection; downloads ~5 GB of GeoTIFFs
python src/download_bands.py
```

This populates `data/bands/` with cloud-filtered Sentinel-2 scenes (2018-2025).

### 3. Run the Full Pipeline

```bash
jupyter notebook notebooks/thesis_pipeline.ipynb
```

Execute all cells in order. The pipeline performs:

1. **Data loading** — reads GeoTIFFs, computes NDMI, applies cloud masking
2. **Exploratory analysis** — generates figures 1-7 (study area, temporal coverage, NDMI time series, inside/outside comparison, change maps)
3. **Dataset construction** — extracts 128x128 patches with sliding window
4. **Model training** — trains ConvLSTM2D with early stopping (generates fig 9)
5. **Single-step prediction** — full-frame NDMI prediction (fig 10)
6. **MC Dropout inference** — 10 stochastic passes for uncertainty (figs 11, 14, 15)
7. **Deforestation severity mapping** — Sobel gradient analysis + X-markers (fig 12)
8. **Multi-step prediction** — 5-step autoregressive rollout (figs 13b, 14)

### 4. Use Pre-trained Weights (Skip Training)

The trained model weights are included at `models/model.pt`. To skip training and go directly to prediction, run all cells in the pipeline notebook but skip the training cell (cell with `for epoch in range(MAX_EPOCHS)`). The model loading cell will pick up the saved weights automatically.


## Requirements

```
numpy>=1.24
torch>=2.0
rasterio>=1.3
matplotlib>=3.7
scipy>=1.10
shapely>=2.0
pyproj>=3.5
jupyter
```
