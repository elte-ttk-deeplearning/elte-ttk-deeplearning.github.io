# add a new metric: greatest uncertainty among correct predictions
# 10000/10000 - 0s - loss: 0.3353 - accuracy: 0.8916 - uncertainty: 0.2304


import tensorflow as tf

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# calculate loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

class uncertainty(tf.keras.metrics.Metric):
    def __init__(self, *kwargs):
        super(uncertainty, self).__init__(*kwargs)
        self.uncertainty = tf.Variable(1.)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        max_true = tf.math.argmax(y_true, axis=1)
        max_pred = tf.math.argmax(y_pred, axis=1)
        true_positions = tf.cast(tf.equal(max_true, max_pred), 'float32')
        uncertainty = tf.cast(1-tf.math.reduce_max(y_pred, axis=1), 'float32')
        self.uncertainty.assign(tf.reduce_max(uncertainty * true_positions))

    def result(self):
        return self.uncertainty

# compile model
model.compile(optimizer='adam',
              loss = loss_fn,
              metrics=['accuracy', uncertainty()])


# optimize parameters to fit model
model.fit(x_train, y_train, epochs=20)

# evaluate model
model.evaluate(x_test, y_test, verbose=2)
