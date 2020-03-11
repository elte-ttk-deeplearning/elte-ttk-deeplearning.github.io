# inputs are pairs of images, output is 1 (same class) or 0 (different class)
# we use the functional api
# 2014/2014 - 0s - loss: 0.2662 - accuracy: 0.9081

import numpy as np
import tensorflow as tf

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_pairs(x, y):
    perm = np.random.permutation(len(x))
    x2 = x[perm]
    y2 = y[perm]
    y_comp = (y == y2).astype(int)

    x_pos = x[y_comp==1]
    x_pos2 = x2[y_comp==1]
    x_neg = x[y_comp==0]
    x_neg2 = x2[y_comp==0]
    pos_cnt = len(x_pos)
    neg_cnt = len(x_neg)
    cnt = np.minimum(pos_cnt, neg_cnt)
    x_pos = x_pos[:cnt]
    x_pos2 = x_pos2[:cnt]
    x_neg = x_neg[:cnt]
    x_neg2 = x_neg2[:cnt]

    input1 = np.concatenate([x_pos, x_neg])
    input2 = np.concatenate([x_pos2, x_neg2])
    target = np.concatenate([np.ones(cnt), np.zeros(cnt)])

    return (input1, input2, target)

x_train, x_train2, y_train = create_pairs(x_train, y_train)
x_test, x_test2, y_test = create_pairs(x_test, y_test)

print("x_train: ", x_train.shape)
print("x_train2: ", x_train2.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("x_test2: ", x_test2.shape)
print("y_test: ", y_test.shape)


# build model
embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='relu')
])

first_input = tf.keras.Input(shape=(28,28), name="first")
second_input = tf.keras.Input(shape=(28,28), name="second")

first_embedding = embedding_model(first_input)
second_embedding = embedding_model(second_input)
embedding = tf.keras.layers.Concatenate(axis=1)([first_embedding, second_embedding])

predictor_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
prediction = predictor_model(embedding)

model = tf.keras.Model(inputs=[first_input, second_input], outputs=prediction)


model.summary()

# calculate loss
loss_fn = tf.keras.losses.BinaryCrossentropy()

# compile model
model.compile(optimizer='adam',
              loss = loss_fn,
              metrics=['accuracy'])


# optimize parameters to fit model
model.fit({'first':x_train, 'second':x_train2}, y_train, epochs=20)

# evaluate model
model.evaluate({'first':x_test, 'second':x_test2}, y_test, verbose=2)
