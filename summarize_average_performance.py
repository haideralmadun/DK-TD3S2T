import hydroeval as he
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)

def relative_root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.mean(y_true) + 1e-8)

def summarize_average_performance(name, y_test_unscaled, y_pred):
    metrics = {
        "RMSE": [], "MAE": [], "R2": [], "MAPE": [], "NSE": [], "PBIAS": [], "SMAPE": [], "RRMSE": [], "KGE": []
    }
    
    for t in range(6):
        y_true, y_hat = y_test_unscaled[:, t], y_pred[:, t]
        metrics["RMSE"].append(np.sqrt(mean_squared_error(y_true, y_hat)))
        metrics["MAE"].append(mean_absolute_error(y_true, y_hat))
        metrics["R2"].append(r2_score(y_true, y_hat))
        metrics["MAPE"].append(mean_absolute_percentage_error(y_true, y_hat))
        metrics["NSE"].append(he.evaluator(he.nse, y_true, y_hat)[0])
        metrics["PBIAS"].append(he.evaluator(he.pbias, y_true, y_hat)[0])
        metrics["SMAPE"].append(np.mean(symmetric_mean_absolute_percentage_error(y_true, y_hat)))
        metrics["RRMSE"].append(relative_root_mean_squared_error(y_true, y_hat))
        metrics["KGE"].append(he.evaluator(he.kge, y_true, y_hat)[0])
        
        # Ensure that we are printing scalar values
        print(f"{name} (T+{t+1}) - RMSE: {metrics['RMSE'][-1]:.4f}, MAE: {metrics['MAE'][-1]:.4f}, R2: {metrics['R2'][-1]:.4f}, ")
        print(f"MAPE: {metrics['MAPE'][-1]:.4f}, NSE: {metrics['NSE'][-1]:.4f}, PBIAS: {metrics['PBIAS'][-1]:.4f}, ")
        
        # Convert arrays to scalars for correct formatting
        smape_value = float(np.mean(metrics["SMAPE"][-1])) if isinstance(metrics["SMAPE"][-1], np.ndarray) else metrics["SMAPE"][-1]
        rrmse_value = float(np.mean(metrics["RRMSE"][-1])) if isinstance(metrics["RRMSE"][-1], np.ndarray) else metrics["RRMSE"][-1]
        kge_value = float(metrics["KGE"][-1]) if isinstance(metrics["KGE"][-1], np.ndarray) else metrics["KGE"][-1]
        
        print(f"SMAPE: {smape_value:.4f}, RRMSE: {rrmse_value:.4f}, KGE: {kge_value:.4f}")
    
    for key in metrics:
        print(f"{name} (Average {key}): {np.mean(metrics[key]):.4f}")
