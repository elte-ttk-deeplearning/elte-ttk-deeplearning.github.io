# numpy example

# sources:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import numpy as np
import time
import math

BATCH_SIZE=64
LEARNING_RATE=1e-6
EPOCHS=200

###############################################
##### load data

x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

train_size = int(0.8*len(x))
val_size = int(0.1*len(x))
test_size = len(x) - train_size - val_size

### note that this train-val-test separation checks extrapolation!
x_train = x[:train_size]
y_train = y[:train_size]

x_val = x[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

x_test = x[train_size+val_size:]
y_test = y[train_size+val_size:]

def dataloader(x,y, batch_size):
    perm = np.random.permutation(len(x))
    x = x[perm]
    y = y[perm]
    for i in range(len(x) // batch_size):
        yield x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]

###############################################
##### specify model computation
# y = a + b*x + c*x^2 + d*x^3

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

# input has shape (BATCH_SIZE, 1, 28, 28)
def model(x):
    y_pred = a + b*x + c * (x**2) + d * (x**3)
    return y_pred

###############################################
##### specify loss function

def loss_fun(y_pred, y):
    return np.square(y_pred - y).sum()

###############################################
##### gradient computation

def grad_fun(x, y_pred, y):
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    return (grad_a, grad_b, grad_c, grad_d)


###############################################
##### training loop

T0 = time.time()
for e in range(EPOCHS):
    train_loss = 0
    trainloader = dataloader(x_train, y_train, BATCH_SIZE)
    for x, y in trainloader:
        y_pred = model(x)        
        train_loss += loss_fun(y_pred, y)
        
        # update weights
        (grad_a, grad_b, grad_c, grad_d) = grad_fun(x, y_pred, y)
        a -= LEARNING_RATE * grad_a
        b -= LEARNING_RATE * grad_b
        c -= LEARNING_RATE * grad_c
        d -= LEARNING_RATE * grad_d

    val_loss = 0
    valloader = dataloader(x_val, y_val, BATCH_SIZE)
    for x, y in valloader:
        y_pred = model(x)
        val_loss += loss_fun(y_pred, y)

    print(f'Train loss: {train_loss/train_size}, val loss: {val_loss/val_size}')

T1 = time.time()
print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
print(f'Training took {T1-T0} sec')

###############################################
##### evaluation

test_loss = 0
testloader = dataloader(x_test, y_test, BATCH_SIZE)
for x, y in testloader:
    y_pred = model(x)
    test_loss += loss_fun(y_pred, y)

print(f'Test loss: {test_loss/test_size}')
