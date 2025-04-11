import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the dataset
df = pd.read_csv("../../data/raw/final_training_data.csv")
# df = pd.read_csv("../../data/raw/more_output.csv")
# df = pd.read_csv("../../data/raw/mod_data.csv")

# Shuffling the dataframe
df = df.sample(frac=1)

# graphical representation of data
sns.countplot(data=df, x="Label")

X = df[df.columns[:7]]
Y = df["Label"]

# Train Test split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)  # random_state = 0 to have same splits across different executions

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error


Y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy using Logistic Regression : ", accuracy)
print(classification_report(Y_test, Y_pred))
roc_curve(Y_test, Y_pred)

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Squarred Error: ", mse)
print("Root Mean Squarred Error: ", rmse)

# DNN Model

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

from keras._tf_keras.keras.callbacks import Callback


# Evaluation
def recall_function(true_positive, false_negative):
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0

    return recall


def precision_function(true_positive, false_positive):
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0

    return precision


def F1_score_function(precision, recall):
    F1_score = (2 * precision * recall) / (precision + recall)

    return F1_score


def accuracy_function(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy


def confusion_matrix(truth, predicted):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for true, pred in zip(truth, predicted):
        if true == 1:
            if pred == true:
                true_positive += 1
            elif pred != true:
                false_negative += 1

        elif true == 0:
            if pred == true:
                true_negative += 1
            elif pred != true:
                false_positive += 1

    accuracy = accuracy_function(
        true_positive, true_negative, false_positive, false_negative
    )
    precision = precision_function(true_positive, false_positive)
    recall = recall_function(true_positive, false_negative)
    return (accuracy, precision, recall)


class MetricsLogger(Callback):
    def __init__(self):
        super().__init__()
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(X_test) > 0.35).astype(int)
        accuracy, precision, recall = confusion_matrix(Y_test, y_pred)
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.f1_list.append(f1)


metrics_logger = MetricsLogger()

# Model
DNN = model.fit(
    X_train,
    Y_train,
    epochs=500,
    verbose=True,
    validation_data=(X_test, Y_test),
    # callbacks=[metrics_logger],
    batch_size=15,
)

training_loss = DNN.history["loss"]
validation_loss = DNN.history["val_loss"]
training_accuracy = DNN.history["accuracy"]
validation_accuracy = DNN.history["val_accuracy"]
training_mse = DNN.history["mean_squared_error"]
validation_mse = DNN.history["val_mean_squared_error"]

precision_list = metrics_logger.precision_list
recall_list = metrics_logger.recall_list
f1_list = metrics_logger.f1_list

epochs = np.arange(1, len(training_loss) + 1)
plt.figure(figsize=(12, 8))

# loss
plt.subplot(2, 2, 1)
plt.plot(epochs, training_loss, label="Training Loss", color="blue")
plt.plot(epochs, validation_loss, label="Validation Loss", color="red")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# accuracy
plt.subplot(2, 2, 2)
plt.plot(epochs, training_accuracy, label="Training Accuracy", color="blue")
plt.plot(epochs, validation_accuracy, label="Validation Accuracy", color="red")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# precision
plt.subplot(2, 2, 3)
plt.plot(epochs, precision_list, label="Precision", color="green")
plt.title("Precision")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend()

# f1-score
plt.subplot(2, 2, 4)
plt.plot(epochs, f1_list, label="F1 Score", color="purple")
plt.title("F1 Score")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.legend()

# mse
plt.subplot(2, 2, 2)
plt.plot(epochs, training_mse, label="Training MSE", color="blue")
plt.plot(epochs, validation_mse, label="Validation MSE", color="red")
plt.title("Mean Squared Error")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend()

# rmse
plt.subplot(2, 2, 2)
plt.plot(epochs, np.sqrt(training_mse), label="Training RMSE", color="blue")
plt.plot(epochs, np.sqrt(validation_mse), label="Validation RMSE", color="red")
plt.title("Root Mean Squared Error")
plt.xlabel("Epochs")
plt.ylabel("Root Mean Squared Error")
plt.legend()


plt.tight_layout()
plt.show()

y_pred_prob = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
auc = roc_auc_score(Y_test, y_pred_prob)

y_pred_prob = lr_model.predict(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
auc = roc_auc_score(Y_test, y_pred_prob)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"DNN (AUC = {auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)", color="red")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()

# Prediction
prediction = model.predict(X_test)


length = len(prediction)
print(prediction)
for i in range(length):
    if prediction[i] >= 0.5:
        prediction[i] = 1
    else:
        prediction[i] = 0

# Evaluation
print(prediction)

accuracy, precision, recall = confusion_matrix(Y_test, prediction)
f1 = F1_score_function(precision, recall)
print("Accuracy : ", accuracy)
print("Precision : ", precision)
print("Recall : ", recall)
print("F1 Score : ", f1)


print(classification_report(Y_test, prediction))

model.save("../../models/landslide_model.h5")
