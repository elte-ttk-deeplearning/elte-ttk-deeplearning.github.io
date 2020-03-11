# https://www.tensorflow.org/tutorials/quickstart/beginner
# kernel 10000/10000 - 0s - loss: 0.4082 - accuracy: 0.8731
# bias 10000/10000 - 0s - loss: 0.3697 - accuracy: 0.8718
# activation 10000/10000 - 0s - loss: 0.3906 - accuracy: 0.8733

import tensorflow as tf

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu',
                          # kernel_regularizer=tf.keras.regularizers.l2(l=0.0001)
                          # bias_regularizer=tf.keras.regularizers.l2(l=0.001)
                          activity_regularizer=tf.keras.regularizers.l2(l=0.001)
    ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax'
    )
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
