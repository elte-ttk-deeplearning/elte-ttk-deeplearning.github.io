# automatic differentiation

# sources:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import numpy as np
import time
import math
import torch

# use gup if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float


BATCH_SIZE=64
LEARNING_RATE=1e-6
EPOCHS=200

###############################################
##### load data 

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

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
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

# input has shape (BATCH_SIZE, 1, 28, 28)
def model(x):
    y_pred = a + b*x + c * (x**2) + d * (x**3)
    return y_pred

###############################################
##### specify loss function

def loss_fun(y_pred, y):
    return (y_pred - y).pow(2).sum()

###############################################
##### training loop

T0 = time.time()
for e in range(EPOCHS):
    train_loss = 0
    trainloader = dataloader(x_train, y_train, BATCH_SIZE)
    for x, y in trainloader:
        y_pred = model(x)
        loss = loss_fun(y_pred, y)
        train_loss += loss.item()        
        
        # update weights
        loss.backward()
        with torch.no_grad():
            a -= LEARNING_RATE * a.grad
            b -= LEARNING_RATE * b.grad
            c -= LEARNING_RATE * c.grad
            d -= LEARNING_RATE * d.grad

            # Manually zero the gradients after updating weights
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

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
