# https://www.tensorflow.org/tutorials/quickstart/beginner
# 10000/10000 - 0s - loss: 0.3705 - accuracy: 0.8650


import tensorflow as tf

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# calculate loss
# cross entropy: - \sum y_target_i * log(y_pred_i)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# compile model
model.compile(optimizer='adam',
              loss = loss_fn,
              metrics=['accuracy'])


# optimize parameters to fit model
model.fit(x_train, y_train, epochs=5)

# evaluate model
model.evaluate(x_test, y_test, verbose=2)
