import pandas as pd
import numpy as np
import seaborn as sns

# Reading the dataset
# df = pd.read_csv("../../data/raw/data.csv")
df = pd.read_csv("../../data/raw/more_output.csv")
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
    X, Y, test_size=0.2, random_state=0
)  # random_state = 0 to have same splits across different executions

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

Y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy using Logistic Regression : ", accuracy)
print(classification_report(Y_test, Y_pred))

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

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Model
DNN = model.fit(
    X_train,
    Y_train,
    epochs=8,
    verbose=True,
    validation_data=(X_test, Y_test),
    batch_size=15,
)

# Prediction
prediction = model.predict(X_test)

length = len(prediction)
for i in range(length):
    if prediction[i] >= 0.35:
        prediction[i] = 1
    else:
        prediction[i] = 0

# Evaluation


def recall_function(true_positive, false_negative):
    recall = true_positive / (true_positive + false_negative)

    return recall


def precision_function(true_positive, false_positive):
    precision = true_positive / (true_positive + false_positive)

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


accuracy, precision, recall = confusion_matrix(Y_test, prediction)
f1 = F1_score_function(precision, recall)
print("Accuracy : ", accuracy)
print("Precision : ", precision)
print("Recall : ", recall)
print("F1 Score : ", f1)


print(classification_report(Y_test, prediction))

model.save("../../models/landslide_model.h5")
