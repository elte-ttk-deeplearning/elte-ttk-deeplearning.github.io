# add two callbacks
# - early stopping
# - save models
# 5000/5000 - 0s - loss: 0.3329 - accuracy: 0.8882

import tensorflow as tf

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_val = x_test[:5000]
y_val = y_test[:5000]
x_test = x_test[5000:]
y_test = y_test[5000:]

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

# compile model
model.compile(optimizer='adam',
              loss = loss_fn,
              metrics=['accuracy'])

callbacks = []
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5))

class mySaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, frequency, **kwargs):
        self.frequency = frequency
        super(mySaveModelCallback, self).__init__(**kwargs)
    
    def on_epoch_end(self, epoch, logs):
        if (epoch+1) % self.frequency == 0:
            tf.keras.models.save_model(self.model, "model_{}".format(epoch+1))
            print("\nSaving model\n")
callbacks.append(mySaveModelCallback(5))
        

# optimize parameters to fit model
model.fit(x_train, y_train, epochs=50,
          validation_data=(x_val, y_val),
          callbacks=callbacks
)


# evaluate model
model.evaluate(x_test, y_test, verbose=2)
