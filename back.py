# back.py

import joblib
import json
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any, List, Optional

# --- New Imports for Optimization ---
try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Optimized Biofilm Prediction & Optimization API",
    version="2.5.0",
    description="Provides predictions and adaptive optimization for S.aureus biofilm removal, learning from user-provided experimental data."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model Artifacts ---
pipeline = None
mechanistic_params = None
feature_importances = None

try:
    pipeline = joblib.load('optimized_biofilm_model.joblib')
    with open('optimized_results.json', 'r') as f:
        full_results = json.load(f)
    mechanistic_params = full_results['realistic_parameters']
    feature_importances = full_results.get('evaluation_results', {}).get('feature_importance', {})
    logger.info("Optimized model and artifacts loaded successfully.")
except Exception as e:
    logger.error(f"FATAL ERROR: Could not load artifacts. {e}", exc_info=True)


# --- Helper Functions (Synchronized with 4practice.py) ---

def run_mechanistic_kernel(params: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    p = params
    ratios = df[["DspB_ratio", "DNase_I_ratio", "ProK_ratio"]].values
    total = ratios.sum(axis=1, keepdims=True) + 1e-9
    vols = ratios / total * df["Total_Volume"].values.reshape(-1, 1)
    concs = (vols / 1000.0) * p["stock_concentration_mg_ml"] / (df["Total_Volume"] / 1000.0).values.reshape(-1, 1)
    dspb_conc, dnase_conc, prok_conc = concs.T
    ph, temp, rt = df["pH"].values, df["Temperature"].values, df["Reaction_Time"].values
    t_dnase, t_prok = df["DNase_Addition_Time"].values, df["ProK_Addition_Time"].values
    eff_times = {"dspb": rt, "dnase": np.clip(rt - t_dnase, 0, None), "prok": np.clip(rt - t_prok, 0, None)}
    def gaussian(x, mu, sigma): return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    mod_ph_dn = gaussian(ph, p["dnase_i"]["opt_ph"], p["dnase_i"]["ph_width"])
    mod_tm_dn = gaussian(temp, p["dnase_i"]["opt_temp"], p["dnase_i"]["temp_width"])
    overlap = np.clip(rt - np.maximum(t_dnase, t_prok), 0, None)
    decay = np.exp(-p["dnase_i"]["k_deg_prok"] * prok_conc * overlap)
    act_dn = mod_ph_dn * mod_tm_dn * decay
    dspb_ph = np.interp(ph, p["dispersin_b"]["ph_profile"]["x"], p["dispersin_b"]["ph_profile"]["y"])
    dspb_tm = gaussian(temp, p["dispersin_b"]["opt_temp"], p["dispersin_b"]["temp_width"])
    act_dspb = dspb_ph * dspb_tm * p["dispersin_b"]["s_aureus_effectiveness"]
    prok_ph = np.interp(ph, p["proteinase_k"]["ph_profile"]["x"], p["proteinase_k"]["ph_profile"]["y"])
    tk, tok = temp + 273.15, p["proteinase_k"]["opt_temp"] + 273.15
    act_prok = prok_ph * np.exp((p["proteinase_k"]["ea"] / 8.314) * (1 / tok - 1 / tk)) * p["proteinase_k"]["s_aureus_effectiveness"]
    v = {"dspb": dspb_conc * p["dispersin_b"]["kcat"] * act_dspb * eff_times["dspb"], "dnase": dnase_conc * p["dnase_i"]["kcat"] * act_dn * eff_times["dnase"], "prok": prok_conc * p["proteinase_k"]["kcat"] * act_prok * eff_times["prok"]}
    total_v = sum(v.values()) + 1e-9
    removal = 100 * (p["s_aureus_biofilm"]["alpha_dnase"] * v["dnase"] + p["s_aureus_biofilm"]["alpha_prok"] * v["prok"] + p["s_aureus_biofilm"]["alpha_dspb"] * v["dspb"]) / total_v
    return np.clip(removal, 0, 100)

def create_advanced_biofilm_features(df: pd.DataFrame) -> pd.DataFrame:
    df_enhanced = df.copy()
    df_enhanced['DspB_PNAG_activity'] = (df['DspB_ratio'] * np.interp(df['pH'], [5.0, 6.0, 6.5, 8.0, 9.0], [0.3, 0.8, 1.0, 0.4, 0.1]) * np.exp(-((df['Temperature'] - 37)**2) / (2 * 15**2)))
    mg_factor = 1.0
    df_enhanced['DNase_eDNA_activity'] = (df['DNase_I_ratio'] * mg_factor * np.where((df['pH'] >= 6.5) & (df['pH'] <= 8.0), 1.0, np.exp(-((df['pH'] - 7.4)**2) / (2 * 0.6**2))) * np.exp(-((df['Temperature'] - 37)**2) / (2 * 10**2)))
    df_enhanced['ProK_protein_activity'] = (df['ProK_ratio'] * np.interp(df['pH'], [6.0, 7.5, 8.5, 9.5, 11.0], [0.2, 0.6, 1.0, 0.9, 0.5]) * np.exp(-((df['Temperature'] - 55)**2) / (2 * 20**2)))
    synergy_factor = 0.5 * np.exp(-((df['pH'] - 7.0)**2) / (2 * 0.8**2)) * np.exp(-((df['Temperature'] - 37)**2) / (2 * 8**2))
    df_enhanced['DNase_DspB_synergy'] = (df_enhanced['DNase_eDNA_activity'] * df_enhanced['DspB_PNAG_activity'] * synergy_factor)
    if 'biofilm_age_hours' in df_enhanced.columns:
        age_factor = np.exp(-df_enhanced['biofilm_age_hours'] / 72.0)
        for enzyme_col in ['DspB_PNAG_activity', 'DNase_eDNA_activity', 'ProK_protein_activity']:
            df_enhanced[enzyme_col] *= age_factor
    total_enzyme = df['DspB_ratio'] + df['DNase_I_ratio'] + df['ProK_ratio'] + 1e-9
    df_enhanced['Total_enzyme_conc'] = total_enzyme
    df_enhanced['Enzyme_diversity_index'] = -np.sum([(df[col]/total_enzyme) * np.log(df[col]/total_enzyme + 1e-9) for col in ['DspB_ratio', 'DNase_I_ratio', 'ProK_ratio']], axis=0)
    return df_enhanced

def get_enhanced_features_list(df_enhanced: pd.DataFrame) -> List[str]:
    base_features = ["DspB_ratio", "DNase_I_ratio", "ProK_ratio", "Total_Volume", "pH", "Temperature", "Reaction_Time", "DNase_Addition_Time", "ProK_Addition_Time", "Addition_Strategy", "biofilm_age_hours"]
    engineered_features = [col for col in df_enhanced.columns if col not in base_features and col not in ['True_Biofilm_Removal', 'Kernel_Prediction', 'Residual', 'removal_bins', 'enzyme_regime', 'stratify_key', 'exp_group']]
    return base_features + engineered_features

# --- API Data Models ---
class PredictionRequest(BaseModel):
    DspB_ratio: float = Field(..., ge=0)
    DNase_I_ratio: float = Field(..., ge=0)
    ProK_ratio: float = Field(..., ge=0)
    Total_Volume: float = Field(..., ge=10)
    pH: float = Field(..., ge=6.8, le=8.2)
    Temperature: float = Field(..., ge=30, le=42)
    Reaction_Time: float = Field(..., ge=1)
    Addition_Strategy: int = Field(..., ge=0, le=1)
    DNase_Addition_Time: float = Field(..., ge=0)
    ProK_Addition_Time: float = Field(..., ge=0)
    biofilm_age_hours: float = Field(..., ge=12, le=96)

class PredictionResponse(BaseModel):
    mean_prediction: float
    prediction_interval_low: float
    prediction_interval_high: float
    kernel_contribution: float
    residual_contribution: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float

class PriorExperiment(BaseModel):
    inputs: Dict[str, float]
    output: float

class OptimizationRequest(BaseModel):
    fixed_conditions: Dict[str, Any]
    prior_experiments: Optional[List[PriorExperiment]] = None

class OptimizationResponse(BaseModel):
    optimal_DspB_ratio: float
    optimal_DNase_I_ratio: float
    optimal_ProK_ratio: float
    max_predicted_removal: float

# --- Full Prediction Logic ---
def get_full_prediction(df: pd.DataFrame) -> np.ndarray:
    X_enhanced = create_advanced_biofilm_features(df)
    all_features = get_enhanced_features_list(X_enhanced)
    X_enhanced = X_enhanced[all_features]
    kernel_pred = run_mechanistic_kernel(mechanistic_params, df)
    residual_pred, _ = pipeline.predict(X_enhanced, return_std=True)
    return np.clip(kernel_pred + residual_pred, 0, 100)

# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root():
    if all([pipeline, mechanistic_params, feature_importances is not None, SKOPT_AVAILABLE]):
        return {"status": "OK", "model": "Optimized Biofilm Model v2.5.0"}
    raise HTTPException(status_code=503, detail="Model, artifacts, or optimizer not loaded.")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not all([pipeline, mechanistic_params]):
        raise HTTPException(status_code=503, detail="Model not available.")
    try:
        input_df = pd.DataFrame([request.model_dump()])
        X_enhanced = create_advanced_biofilm_features(input_df)
        all_features = get_enhanced_features_list(X_enhanced)
        X_enhanced = X_enhanced[all_features]
        kernel_pred = run_mechanistic_kernel(mechanistic_params, input_df)[0]
        residual_pred, epistemic_std = pipeline.predict(X_enhanced, return_std=True)
        residual_pred = residual_pred[0]
        epistemic_uncertainty = epistemic_std[0]
        gp_model = pipeline.named_steps['regressor']
        aleatoric_uncertainty = np.sqrt(gp_model.kernel_.k2.noise_level)
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        calibration_factor = 1.96 
        adjusted_uncertainty = total_uncertainty * calibration_factor
        mean_pred = np.clip(kernel_pred + residual_pred, 0, 100)
        return PredictionResponse(
            mean_prediction=mean_pred,
            prediction_interval_low=np.clip(mean_pred - adjusted_uncertainty, 0, 100),
            prediction_interval_high=np.clip(mean_pred + adjusted_uncertainty, 0, 100),
            kernel_contribution=kernel_pred,
            residual_contribution=residual_pred,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance", summary="Get Feature Importances")
def get_feature_importance():
    if feature_importances is None:
        raise HTTPException(status_code=404, detail="Feature importance data not available.")
    return feature_importances

@app.post("/optimize-mix", response_model=OptimizationResponse, summary="Find Optimal Enzyme Mix")
def optimize_mix(request: OptimizationRequest):
    if not SKOPT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Optimization feature requires scikit-optimize.")
    
    fixed_params = request.fixed_conditions

    def objective(ratios):
        data = {**fixed_params, 'DspB_ratio': ratios[0], 'DNase_I_ratio': ratios[1], 'ProK_ratio': ratios[2]}
        df = pd.DataFrame([data])
        prediction = get_full_prediction(df)[0]
        return -prediction

    space = [Real(0.1, 10, name='DspB_ratio'),
             Real(0.1, 10, name='DNase_I_ratio'),
             Real(0.1, 10, name='ProK_ratio')]
    
    x0, y0 = None, None
    if request.prior_experiments:
        x0 = [[exp.inputs['DspB_ratio'], exp.inputs['DNase_I_ratio'], exp.inputs['ProK_ratio']] for exp in request.prior_experiments]
        y0 = [-exp.output for exp in request.prior_experiments]
        logger.info(f"Starting optimization with {len(x0)} prior data points.")

    result = gp_minimize(objective, space, n_calls=25, random_state=42, n_jobs=-1, x0=x0, y0=y0)

    optimal_ratios = result.x
    max_removal = -result.fun

    return OptimizationResponse(
        optimal_DspB_ratio=optimal_ratios[0],
        optimal_DNase_I_ratio=optimal_ratios[1],
        optimal_ProK_ratio=optimal_ratios[2],
        max_predicted_removal=max_removal
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
