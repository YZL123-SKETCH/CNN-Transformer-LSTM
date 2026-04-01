import torch
import torch.nn as nn
import os
import numpy as np

from dataset import DataLoader
from Module.CNN_Transformer_LSTM import CNN_Transformer_LSTM
from config import get_args
from metrics import (
    calculate_metrics,
    save_metrics_to_excel,
    plot_training_metrics,
    plot_true_vs_pred,
    save_true_pred_to_excel,
    plot_test_metrics
)

# ===================== 1. Configuration Initialization =====================
args = get_args()
device = args.device
os.makedirs(args.save_file, exist_ok=True)

# ===================== 2. Load Dataset =====================
loader = DataLoader(
    filename=args.filename,
    split_ratio=0.77,
    cols=[ ]
)

# ===================== 3. Model Initialization =====================
model = CNN_Transformer_LSTM(
    input_size=args.input_size,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    output_size=args.output_size,
    dropout=args.dropout
).to(device)

# ===================== 4. Loss Function & Optimizer =====================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ===================== 5. Metric Recording =====================
train_metrics_history = {
    "epoch": [], "loss": [], "MSE": [], "MAE": [], "RMSE": [], "R2": []
}
test_metrics_history = {
    "epoch": [], "MSE": [], "MAE": [], "RMSE": [], "R2": []
}

best_r2 = -float("inf")
best_epoch = 0

# ===================== 6. Main Training Loop =====================
for epoch in range(1, args.epochs + 1):
    model.train()
    total_train_loss = 0.0
    batch_count = 0

    train_targets = []
    train_predictions = []

    # Iterate over training batches
    for x_rut, x_temp, x_load, y_true_batch in loader.generate_train_batch(args.sequence_length, args.batch_size):
        # Convert numpy array to PyTorch tensor
        rut_seq = torch.tensor(x_rut, dtype=torch.float32).unsqueeze(-1).to(device)
        temp_seq = torch.tensor(x_temp, dtype=torch.float32).unsqueeze(-1).to(device)
        load_seq = torch.tensor(x_load, dtype=torch.float32).unsqueeze(-1).to(device)
        y_batch = torch.tensor(y_true_batch, dtype=torch.float32).unsqueeze(-1).to(device)

        # Forward propagation
        optimizer.zero_grad()
        y_pred_batch = model(rut_seq, temp_seq, load_seq)
        loss = criterion(y_pred_batch, y_batch)

        # Backward propagation & optimization
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        batch_count += 1

        # Collect results
        train_targets.append(y_batch.detach().cpu().numpy())
        train_predictions.append(y_pred_batch.detach().cpu().numpy())

    # Calculate average training loss
    avg_train_loss = total_train_loss / batch_count

    # Concatenate all training results
    y_train_true = np.concatenate(train_targets)
    y_train_pred = np.concatenate(train_predictions)

    # Inverse normalization for raw value recovery
    dummy_true_train = np.zeros((y_train_true.shape[0], 3))
    dummy_pred_train = np.zeros((y_train_pred.shape[0], 3))
    dummy_true_train[:, 0] = y_train_true[:, 0]
    dummy_pred_train[:, 0] = y_train_pred[:, 0]

    inv_y_train_true = loader.scaler.inverse_transform(dummy_true_train)[:, 0]
    inv_y_train_pred = loader.scaler.inverse_transform(dummy_pred_train)[:, 0]

    # Compute training evaluation metrics
    train_mse, train_mae, train_rmse, train_r2 = calculate_metrics(inv_y_train_true, inv_y_train_pred)

    # Save training metrics
    train_metrics_history["epoch"].append(epoch)
    train_metrics_history["loss"].append(avg_train_loss)
    train_metrics_history["MSE"].append(train_mse)
    train_metrics_history["MAE"].append(train_mae)
    train_metrics_history["RMSE"].append(train_rmse)
    train_metrics_history["R2"].append(train_r2)

    # Print training log
    print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_train_loss:.6f} | MSE: {train_mse:.6f} | MAE: {train_mae:.6f} | RMSE: {train_rmse:.6f} | R2: {train_r2:.6f}")

    # ===================== 7. Test Evaluation Every Epoch =====================
    model.eval()
    test_targets = []
    test_predictions = []

    with torch.no_grad():
        for x_rut, x_temp, x_load, y_true_batch in loader.generate_test_batch(args.sequence_length, args.batch_size):
            rut_seq_test = torch.tensor(x_rut, dtype=torch.float32).unsqueeze(-1).to(device)
            temp_seq_test = torch.tensor(x_temp, dtype=torch.float32).unsqueeze(-1).to(device)
            load_seq_test = torch.tensor(x_load, dtype=torch.float32).unsqueeze(-1).to(device)
            y_test_batch = torch.tensor(y_true_batch, dtype=torch.float32).to(device)

            y_pred_test_batch = model(rut_seq_test, temp_seq_test, load_seq_test)

            test_targets.append(y_test_batch.detach().cpu().numpy())
            test_predictions.append(y_pred_test_batch.detach().cpu().numpy())

    # Concatenate test results
    y_test_true = np.concatenate(test_targets).reshape(-1, 1)
    y_test_pred = np.concatenate(test_predictions).reshape(-1, 1)

    # Inverse normalization
    dummy_true_test = np.zeros((y_test_true.shape[0], 3))
    dummy_pred_test = np.zeros((y_test_pred.shape[0], 3))
    dummy_true_test[:, 0] = y_test_true[:, 0]
    dummy_pred_test[:, 0] = y_test_pred[:, 0]

    inv_y_test_true = loader.scaler.inverse_transform(dummy_true_test)[:, 0]
    inv_y_test_pred = loader.scaler.inverse_transform(dummy_pred_test)[:, 0]

    # Compute test metrics
    test_mse, test_mae, test_rmse, test_r2 = calculate_metrics(inv_y_test_true, inv_y_test_pred)

    # Save test metrics
    test_metrics_history["epoch"].append(epoch)
    test_metrics_history["MSE"].append(test_mse)
    test_metrics_history["MAE"].append(test_mae)
    test_metrics_history["RMSE"].append(test_rmse)
    test_metrics_history["R2"].append(test_r2)

    # Update best model record
    if test_r2 > best_r2:
        best_r2 = test_r2
        best_epoch = epoch

    # Print test log
    print(f"Epoch {epoch} Test | MSE: {test_mse:.6f} | MAE: {test_mae:.6f} | RMSE: {test_rmse:.6f} | R2: {test_r2:.6f}")

# ===================== 8. Output Best Overall Result =====================
best_idx = np.argmax(test_metrics_history["R2"])
final_best_epoch = test_metrics_history["epoch"][best_idx]
final_best_mse = test_metrics_history["MSE"][best_idx]
final_best_mae = test_metrics_history["MAE"][best_idx]
final_best_rmse = test_metrics_history["RMSE"][best_idx]
final_best_r2 = test_metrics_history["R2"][best_idx]

print("\n========== Best Test Performance ==========")
print(f"Best Epoch: {final_best_epoch}")
print(f"MSE: {final_best_mse:.6f}")
print(f"MAE: {final_best_mae:.6f}")
print(f"RMSE: {final_best_rmse:.6f}")
print(f"R2: {final_best_r2:.6f}")

# ===================== 9. Save All Results =====================
result_dir = args.save_file
os.makedirs(result_dir, exist_ok=True)

# Save metric Excel files
save_metrics_to_excel(train_metrics_history, os.path.join(result_dir, "train_metrics.xlsx"))
save_metrics_to_excel(test_metrics_history, os.path.join(result_dir, "test_metrics.xlsx"))

# Save metric figures
plot_training_metrics(train_metrics_history, os.path.join(result_dir, "train_metrics_plot.png"))
plot_test_metrics(test_metrics_history, os.path.join(result_dir, "test_metrics_plot.png"))

# Save true vs predicted values & figures at final epoch
plot_true_vs_pred(inv_y_train_true, inv_y_train_pred, os.path.join(result_dir, "train_true_vs_pred.png"))
plot_true_vs_pred(inv_y_test_true, inv_y_test_pred, os.path.join(result_dir, "test_true_vs_pred.png"))

save_true_pred_to_excel(inv_y_train_true, inv_y_train_pred, os.path.join(result_dir, "train_true_vs_pred.xlsx"))
save_true_pred_to_excel(inv_y_test_true, inv_y_test_pred, os.path.join(result_dir, "test_true_vs_pred.xlsx"))

print("\nAll results saved successfully!")










