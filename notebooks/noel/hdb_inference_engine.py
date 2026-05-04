# %% [code]
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def run_ensemble_prediction(X_processed, model_dir, num_folds=5):
    logs = []
    print("run_ensemble_prediction...", flush=True)
    logs.append("run_ensemble_prediction...")
    """
    Loads 5-fold models from the registry and returns averaged predictions.
    """
    # Initialize an array for the log-transformed predictions
    print("run_ensemble_prediction: Initialize an array for the log-transformed predictions...", flush=True)
    logs.append("run_ensemble_prediction: Initialize an array for the log-transformed predictions...")
    final_preds_log = np.zeros(len(X_processed))
    
    for fold in range(num_folds):
        print("run_ensemble_prediction: Loading Fold {fold} for prediction...", flush=True)
        logs.append("Loading Fold {fold} for prediction...")
        model = CatBoostRegressor()
        
        # This path will point to your HDB-Price-Predictor asset
        model_file_path = f"{model_dir}/hdb_model_fold_{fold}.cbm"
        print(f"run_ensemble_prediction: HDB-Price-Predictor asset: {model_file_path}...", flush=True)
        logs.append(f"run_ensemble_prediction: HDB-Price-Predictor asset: {model_file_path}...")
        model.load_model(model_file_path)
        
        # Add this fold's prediction to the total (ensemble averaging)
        print("run_ensemble_prediction: Add this fold's prediction to the total (ensemble averaging)...", flush=True)
        logs.append("run_ensemble_prediction: Add this fold's prediction to the total (ensemble averaging)...")
        final_preds_log += model.predict(X_processed) / num_folds
        
    # ── Post-Processing ──────────────────────────────────────
    # 1. Reverse the log1p transform used during training
    # 2. Apply safety clipping to keep prices within reasonable HDB bounds
    print("run_ensemble_prediction: Post-Processing...", flush=True)
    logs.append("run_ensemble_prediction: Post-Processing...")
    final_prices = np.expm1(final_preds_log).clip(100000, 1300000)

    status_report = {
        'messages': logs,
        'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
    }
    
    return final_prices, status_report