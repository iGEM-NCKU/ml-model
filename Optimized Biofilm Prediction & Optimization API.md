<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Optimized Biofilm Prediction \& Optimization API

This project is an advanced machine learning and mechanistic modeling system for **predicting and optimizing S. aureus biofilm removal** using a multi-enzyme cocktail (Dispersin B, DNase I, Proteinase K). It includes a **FastAPI backend** (`back.py`) with model inference and optimization endpoints, and a **frontend HTML dashboard** (`frontend.html`) for experiment analysis and visualization.

## Table of Contents

- [Features](#features)
- [System Overview](#system-overview)
- [Requirements](#requirements)
- [Usage](#usage)
    - [1. Backend: Setting Up and Running](#1-backend-setting-up-and-running)
    - [2. API Endpoints](#2-api-endpoints)
    - [3. Frontend Dashboard](#3-frontend-dashboard)
- [Key Concepts](#key-concepts)
- [Code Structure](#code-structure)
- [Extending and Customizing](#extending-and-customizing)


## Features

- **Biofilm Removal Prediction**: Given input experiment parameters, predicts biofilm removal efficiency.
- **Enzyme Cocktail Optimization**: Suggests the optimal mix of Dispersin B, DNase I, and Proteinase K for maximum biofilm removal under fixed experimental conditions.
- **Robust Kernel + Residual Model**: Combines mechanistic and machine learning (Gaussian process regression) predictions.
- **Uncertainty Quantification**: Provides prediction intervals and separate epistemic/aleatoric uncertainties.
- **Feature Importance Analysis**: Ranks input features by impact on predictions (API endpoint).
- **Frontend Dashboard**: Visualizes predictions, actual results, feature importances, and experiment history.


## System Overview

The platform takes *experimental parameters* (enzyme ratios, pH, temperature, volume, reaction time, etc.), computes engineered features, predicts S. aureus biofilm removal, and can optimize enzyme ratios for best performance. It allows tracking and analysis of actual/experiment results versus predictions.

**Mechanistic Model**: Encodes literature and empirical enzyme characteristics (pH, temperature, synergies, etc.)
**Machine Learning Model**: Gaussian process regressor trained on biofilm removal data for data-driven improvements.

## Requirements

- Python 3.8+
- pip
- Required packages:
    - fastapi
    - scikit-learn
    - numpy
    - pandas
    - scikit-optimize
    - uvicorn
    - joblib
    - pydantic

For interactive dashboard, an HTML-friendly browser.

## Usage

### 1. Backend: Setting Up and Running

#### a. Install dependencies

```bash
pip install fastapi uvicorn scikit-learn pandas numpy scikit-optimize joblib pydantic
```


#### b. Prepare model artifacts

- Place your trained model in `optimized_biofilm_model.joblib`.
- Include `optimized_results.json` with `realistic_parameters` and (optionally) feature importances.


#### c. Run the API server

```bash
python back.py
```

_The API will launch at `127.0.0.1:8000` by default._

### 2. API Endpoints

#### Health Check

- **GET /**
Returns status if the model and components are loaded.


#### Predict Biofilm Removal

- **POST /predict**
Body: All experiment parameters (see `PredictionRequest` class).
Response:
    - Mean predicted removal
    - Prediction interval (low/high)
    - Kernel and residual contributions
    - Epistemic and aleatoric uncertainties


#### Feature Importance

- **GET /feature-importance**
Returns importance ranking for all input features.


#### Optimize Enzyme Mix

- **POST /optimize-mix**
Body: `fixed_conditions` (dict of experimental settings, e.g. pH/temperature/volume/etc.), optional `prior_experiments` for warm start.
Returns optimal ratios and max-predicted removal.


### 3. Frontend Dashboard

Open `frontend.html` in your browser after backend is running.

**Features visible:**

- Set experiment conditions: pH, Temperature (°C), Volume (µL), Biofilm Age (h), Control OD600
- Trigger optimization for best enzyme mix
- Add new experiment runs (track actual/predicted, visualize errors)
- View model accuracy (MAE), feature importances, and input-output trends


## Key Concepts

- **Enzyme Ratios** (`DspB_ratio`, `DNase_I_ratio`, `ProK_ratio`): User-set or optimized, these determine the proportional volumes of each enzyme in the test assay.
- **Engineered Features**: pH- and temperature-modulated enzymatic activity, total enzyme conc., synergy index, etc., are automatically constructed for improved predictions.
- **Mechanistic + ML Model**: Mechanistic kernel (based on literature and physical chemistry) + Gaussian process residual corrector.
- **Uncertainty Quantification**:
    - Epistemic: Model (data-driven) uncertainty
    - Aleatoric: Irreducible noise (intrinsic variability)
    - Prediction interval: Combined range with 95% confidence (approximately mean ± 1.96 × total uncertainty)


## Code Structure

- **back.py**: Backend FastAPI application.
    - Loads model and parameters at startup.
    - Defines all helper functions for feature engineering, kernel predictions, and API endpoint logic.

Key segments:
    - `run_mechanistic_kernel`: Applies mechanistic model to input.
    - `create_advanced_biofilm_features`, `get_enhanced_features_list`: Feature engineering.
    - `get_full_prediction`: Total prediction (mechanistic + ML).
    - `/predict`: Prediction endpoint.
    - `/feature-importance`: Returns model feature importances.
    - `/optimize-mix`: Runs Gaussian process Bayesian optimization for best mix under any fixed settings.
    - Logging, error handling, and CORS for frontend compatibility.
- **frontend.html**: User dashboard.
    - UI fields for all relevant experiment inputs.
    - Buttons to fetch optimal mix, visualize results, and export/import data.
    - Displays model analysis summaries and predictions for each experimental run.


## Extending and Customizing

- **Biofilm/Enzyme System**:
Customize the mechanistic kernel or the trained ML pipeline for different organisms or enzymes.
- **Feature Engineering**:
Add new features to `create_advanced_biofilm_features`, e.g. new synergy or inhibition terms.
- **Model Retraining**:
Use actual experiment datasets to retrain or fine-tune the Gaussian process regressor (`pipeline`).
- **New Endpoints**:
Add more analytics (e.g. batch prediction) via new FastAPI routes.

**If you use or modify this package for your research, cite this repository and any related publications. For bug reports or feature requests, please open an issue or submit a pull request.**

<div style="text-align: center">⁂</div>

[^1]: back.py

[^2]: frontend.html

