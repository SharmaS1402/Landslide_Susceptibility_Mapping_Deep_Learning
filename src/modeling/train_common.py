import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/raw/data.csv")

X = df[df.columns[:7]]
Y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
fpr1, tpr1, thresholds = roc_curve(y_test, y_pred1)
auc1 = roc_auc_score(y_test, y_pred1)


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
y_pred2 = rf_regressor.predict(X_test)
fpr2, tpr2, thresholds = roc_curve(y_test, y_pred2)
auc2 = roc_auc_score(y_test, y_pred2)


svm_regressor = SVR(kernel="rbf", C=1.0, epsilon=0.1)
svm_regressor.fit(X_train, y_train)
y_pred3 = svm_regressor.predict(X_test)
fpr3, tpr3, thresholds = roc_curve(y_test, y_pred3)
auc3 = roc_auc_score(y_test, y_pred3)


lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)
y_pred4 = lr_model.predict(X_test)
fpr4, tpr4, thresholds = roc_curve(y_test, y_pred4)
auc4 = roc_auc_score(y_test, y_pred4)

# DNN

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras import layers
# from keras._tf_keras.keras.layers import Dense

# MODEL ARCHITECTURE
input_dim = X_train.shape[1]
model = Sequential()
model.add(layers.Dense(30, input_dim=input_dim, activation="relu"))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(870, activation="relu"))

model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation="sigmoid"))

# Compiling the model

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy", "mean_squared_error"],
)

DNN = model.fit(
    X_train,
    y_train,
    epochs=500,
    verbose=True,
    validation_data=(X_test, y_test),
    batch_size=15,
)


y_pred6 = model.predict(X_test)
fpr6, tpr6, thresholds = roc_curve(y_test, y_pred6)
auc6 = roc_auc_score(y_test, y_pred6)

label1_data = pd.read_csv("../../data/raw/ahp_label1.csv")
label0_data = pd.read_csv("../../data/raw/ahp_label0_filtered.csv")

# AHP
data = pd.concat([label1_data, label0_data], ignore_index=True)

# Extract true labels and predicted values
y_true = data["label"]
y_pred5 = data["ahp_index"]

y_pred_binary = (y_pred5 >= 0.65).astype(int)

fpr5, tpr5, thresholds = roc_curve(y_true, y_pred_binary)
auc5 = roc_auc_score(y_true, y_pred_binary)


# prediction = rf_regressor.predict(X_test[8:])
# print(prediction)


# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f"GNB (AUC = {auc1:.2f})", color="blue")
plt.plot(fpr2, tpr2, label=f"RF (AUC = {auc2:.2f})", color="purple")
plt.plot(fpr3, tpr3, label=f"SVM (AUC = {auc3:.2f})", color="green")
plt.plot(fpr4, tpr4, label=f"LR (AUC = {auc4:.2f})", color="violet")
plt.plot(fpr5, tpr5, label=f"AHP (AUC = {auc5:.2f})", color="orange")
plt.plot(fpr6, tpr6, label=f"AHP (AUC = {auc6:.2f})", color="red")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)", color="red")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
