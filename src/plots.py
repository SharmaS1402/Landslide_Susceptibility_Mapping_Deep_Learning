from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras import layers
from keras._tf_keras.keras.utils import plot_model

input_dim = 7
model = Sequential()
model.add(layers.Dense(30, input_dim=input_dim, activation="relu"))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(870, activation="relu"))

model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation="sigmoid"))

plot_model(
    model,
    to_file="./model_architecture.png",
    show_shapes=True,  # shows (batch_size, features) info
    show_layer_names=True,
)
