from keras.models import Sequential
from keras.layers import Dense, Flatten, Rescaling
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import image_dataset_from_directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OCCULT_NEURONS = 100
ACTIVATION_FUNCTION = "relu"
LOSS = "mean_squared_error"
OPTIMIZER = "sgd"
INPUT_SHAPE = (496, 369, 3)
CLASSES = 4

# Input dimension: 496x369

# Dataset import
train_ds, val_ds = image_dataset_from_directory(
    directory="./images/waveforms",
    labels="inferred",
    label_mode="categorical",  # uses one of classes
    batch_size=1,
    seed=8631,
    image_size=(496, 369),
    validation_split=0.2,
    subset="both",
)

# Normalization
normalization_layer = Rescaling(1.0 / 255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Creating batchs
train_image_batch, train_labels_batch = next(iter(normalized_train_ds))
val_image_batch, val_labels_batch = next(iter(normalized_val_ds))
print(train_image_batch.shape)

# CNN structure
classifier = Sequential()
classifier.add(
    Conv2D(
        32,
        (4, 3),
        activation=ACTIVATION_FUNCTION,
        padding="same",
        input_shape=INPUT_SHAPE,
    )
)
classifier.add(MaxPooling2D(pool_size=(4, 3)))
classifier.add(Conv2D(32, (4, 3), activation=ACTIVATION_FUNCTION, padding="same"))
classifier.add(MaxPooling2D(pool_size=(4, 3)))
classifier.add(Flatten())  # (x/conv_x/pool_x) * (y/conv_y/pool_y) = input dim
classifier.add(Dense(units=OCCULT_NEURONS, activation=ACTIVATION_FUNCTION))
classifier.add(Dense(units=CLASSES))  # output layer
classifier.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
history = classifier.fit(
    x=normalized_train_ds,
    epochs=30,
    verbose=1,
    validation_data=normalized_val_ds,
)

# Generate result graphs
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history.history["accuracy"], "o-", label="accuracy")
ax1.plot(history.history["val_accuracy"], "o-", label="val_accuracy")
plt.legend()
ax2.plot(history.history["loss"], ".-", label="loss")
ax2.plot(history.history["val_loss"], ".-", label="val_loss")
plt.legend()
plt.savefig("outputs/cnn_wave_results.jpg")

# Generate results table
table = np.array(
    [
        history.history["accuracy"],
        history.history["loss"],
        history.history["val_accuracy"],
        history.history["val_loss"],
    ]
)
df = pd.DataFrame(data=table).T
df.columns = ["accuracy", "loss", "val_accuracy", "val_loss"]
file_name = f"cnn_wave_results.xlsx"
file_path = f"outputs/{file_name}"
df.to_excel(excel_writer=file_path)
