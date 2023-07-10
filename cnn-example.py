from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
import pandas as pd
import numpy as np

OCCULT_NEURONS = 100
ACTIVATION_FUNCTION = "tanh"
LOSS = "mean_squared_error"
OPTIMIZER = "sgd"
BATCH_SIZE = 25

#Database import e format
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_reshaped = x_train.reshape(x_train.shape[0],32,32,3)
x_test_reshaped = x_test.reshape(x_test.shape[0],32,32,3)

x_train_asFloat32 = x_train_reshaped.astype("float32")
x_test_asFloat32 = x_test_reshaped.astype("float32")

x_train_norm = x_train_asFloat32 / 255.0
x_test_norm = x_test_asFloat32 / 255.0

#Targets import
cbvs10 = np.array(np.loadtxt(open("cbvs10.csv", "rb"), 
                                         delimiter=",", skiprows=0), np.float32)

#Target format
t_train = np.zeros((y_train.shape[0],10), dtype=np.float32)
for i in range(y_train.shape[0]):
    t_train[i] = cbvs10[y_train[i]]

t_test = np.zeros((y_test.shape[0],10), dtype=np.float32)
for i in range(y_test.shape[0]):
    t_test[i] = cbvs10[y_test[i]]

#CNN Structure
classifier = Sequential()
classifier.add(Conv2D(32, (3,3), activation = ACTIVATION_FUNCTION, padding = 'same', input_shape=(32,32,3)))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = OCCULT_NEURONS, activation = ACTIVATION_FUNCTION))
classifier.add(Dense(units = 10, activation = ACTIVATION_FUNCTION))
classifier.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = ['accuracy'])
history = classifier.fit(
    x = x_train,
    y = t_train,
    batch_size = BATCH_SIZE,
    epochs = 50,
    verbose = 1,
    validation_data = (x_test, t_test),
)

#Generate results table
table = np.array([history.history['accuracy'], history.history['loss'],history.history['val_accuracy'], history.history['val_loss']])
df = pd.DataFrame(data=table).T
df.columns = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
file_name = f"cnn_neurons=100_act={ACTIVATION_FUNCTION}_loss={LOSS}_opt={OPTIMIZER}_bsize={BATCH_SIZE}.xlsx"
file_path = f"outputs/{file_name}"
df.to_excel(excel_writer = file_path)