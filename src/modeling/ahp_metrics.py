import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_squared_error,
)
import matplotlib.pyplot as plt


# Load the data
label1_data = pd.read_csv("../../data/raw/ahp_label1.csv")
label0_data = pd.read_csv("../../data/raw/ahp_label0_filtered.csv")

# Concatenate the data
data = pd.concat([label1_data, label0_data], ignore_index=True)

# Extract true labels and predicted values
y_true = data["label"]
y_pred = data["ahp_index"]

# Binarize predictions (using 0.5 as threshold)
y_pred_binary = (y_pred >= 0.65).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred_binary)
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)

mse = mean_squared_error(y_true, y_pred_binary)
rmse = np.sqrt(mse)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_binary)
auc = roc_auc_score(y_true, y_pred_binary)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AHP (AUC = {auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)", color="red")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
