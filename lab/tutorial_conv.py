# adding some convolutional layers
# 10000/10000 - 1s - loss: 0.2481 - accuracy: 0.9100

import numpy as np
import tensorflow as tf

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(10, (3,3), padding='valid', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(50, (4,4), padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(100, (3,3), padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()



# calculate loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# compile model
model.compile(optimizer='adam',
              loss = loss_fn,
              metrics=['accuracy'])

# optimize parameters to fit model
model.fit(x_train, y_train, epochs=5)

# evaluate model
model.evaluate(x_test, y_test, verbose=2)
