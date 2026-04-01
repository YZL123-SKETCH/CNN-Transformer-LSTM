import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics
    Returns: mse, mae, rmse, r2
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, r2


def save_metrics_to_excel(metrics_history, save_path):
    """Save training history metrics to Excel"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(metrics_history)
    df.to_excel(save_path, index=False)


def plot_training_metrics(metrics_history, save_path):
    """Plot training loss and evaluation metrics"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history["epoch"], metrics_history["loss"], label='Loss', linewidth=1.5)
    plt.plot(metrics_history["epoch"], metrics_history["MSE"], label='MSE', linewidth=1.5)
    plt.plot(metrics_history["epoch"], metrics_history["MAE"], label='MAE', linewidth=1.5)
    plt.plot(metrics_history["epoch"], metrics_history["RMSE"], label='RMSE', linewidth=1.5)
    plt.plot(metrics_history["epoch"], metrics_history["R2"], label='R²', linewidth=1.5)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Training Metrics", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_test_metrics(metrics_history, save_path):
    """Plot test metrics only"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history["epoch"], metrics_history["MSE"], label='MSE', linewidth=1.5)
    plt.plot(metrics_history["epoch"], metrics_history["MAE"], label='MAE', linewidth=1.5)
    plt.plot(metrics_history["epoch"], metrics_history["RMSE"], label='RMSE', linewidth=1.5)
    plt.plot(metrics_history["epoch"], metrics_history["R2"], label='R²', linewidth=1.5)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Test Metrics", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_true_vs_pred(y_true, y_pred, save_path):
    """Plot true values vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True', color='#1f77b4', linewidth=1.5)
    plt.plot(y_pred, label='Predicted', color='#ff4b5c', linewidth=1.5, alpha=0.8)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("True vs Predicted", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_true_pred_to_excel(y_true, y_pred, save_path):
    """Save true and predicted values to Excel"""
    df = pd.DataFrame({
        "True": y_true.flatten(),
        "Predicted": y_pred.flatten()
    })
    df.to_excel(save_path, index=False)