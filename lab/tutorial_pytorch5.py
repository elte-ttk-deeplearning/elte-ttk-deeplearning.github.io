# fashion mnist with optimizer

# sources:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

import numpy as np
import time
import math
import torch
from torchvision import datasets, transforms
from torch import nn, optim

# use gup if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float


BATCH_SIZE=256
LEARNING_RATE=1e-2
MOMENTUM=0.9
EPOCHS=15

INPUT_SIZE = 784
HIDDEN_SIZES = [128, 64]
OUTPUT_SIZE = 10


###############################################
##### load data 

# define data preprocessing transformations
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
])

# create a data loader
trainset = datasets.FashionMNIST('data', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('data', download=True, train=False, transform=transform)

train_size = int(0.8*len(trainset))
val_size = len(trainset) - train_size
test_size = len(testset)
trainset, valset = torch.utils.data.random_split(trainset, (train_size, val_size))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

###############################################
##### specify model computation
# 3 layer neural network with ReLU nonlinearity

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0]),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE),
    nn.LogSoftmax(dim=1)
)
print(model)
model.to(device)

###############################################
##### specify loss function
# negative log likelihood loss

loss_fun = nn.NLLLoss()

def success_count(log_probs, labels):
    success = torch.argmax(log_probs, dim=1) == labels
    return success.sum()


###############################################
##### training loop

# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
# optimizer = optim.Adadelta(model.parameters())
# optimizer = optim.Adagrad(model.parameters())
# optimizer = optim.Adam(model.parameters())
optimizer = optim.AdamW(model.parameters())
# optimizer = optim.RMSprop(model.parameters())
# optimizer = optim.Rprop(model.parameters())

T0 = time.time()
for e in range(EPOCHS):
    train_loss = 0
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        log_probs = model(images)

        loss = loss_fun(log_probs, labels) # negative log likelihood loss
        train_loss += loss.item()
        
        # update weights
        optimizer.zero_grad() # Zero the gradients before running the backward pass
        loss.backward()

        optimizer.step()


    val_loss = 0
    val_success = 0
    for images, labels in valloader:
        images = images.to(device)
        labels = labels.to(device)
        log_probs = model(images)

        loss = loss_fun(log_probs, labels) # negative log likelihood loss
        val_loss += loss.item()
        val_success += success_count(log_probs, labels).item()

    print(f'Epoch {e}: Train loss: {train_loss/train_size}, val loss: {val_loss/val_size}, val acc: {val_success/val_size:.2f}')

T1 = time.time()
    
print(f'Training took {T1-T0} sec')

###############################################
##### evaluation

test_loss = 0
test_success = 0 
for images, labels in testloader:
    images = images.to(device)
    labels = labels.to(device)
    log_probs = model(images)

    loss = loss_fun(log_probs, labels) # negative log likelihood loss
    test_loss += loss.item()
    test_success += success_count(log_probs, labels).item()

print(f'Test loss: {test_loss/test_size}, test acc: {test_success/test_size:.2f}')

# Sequential(
#   (0): Flatten(start_dim=1, end_dim=-1)
#   (1): Linear(in_features=784, out_features=128, bias=True)
#   (2): ReLU()
#   (3): Linear(in_features=128, out_features=64, bias=True)
#   (4): ReLU()
#   (5): Linear(in_features=64, out_features=10, bias=True)
#   (6): LogSoftmax(dim=1)
# )
# Epoch 0: Train loss: 0.002563951143374046, val loss: 0.0019286951273679734, val acc: 0.82
# Epoch 1: Train loss: 0.001677922220279773, val loss: 0.001655953879157702, val acc: 0.85
# Epoch 2: Train loss: 0.0015075828700015941, val loss: 0.0015063098967075348, val acc: 0.86
# Epoch 3: Train loss: 0.001395273121073842, val loss: 0.0015449144815405209, val acc: 0.86
# Epoch 4: Train loss: 0.0013310163424660763, val loss: 0.0014644139309724171, val acc: 0.86
# Epoch 5: Train loss: 0.0012482098148514826, val loss: 0.0014740744332472483, val acc: 0.86
# Epoch 6: Train loss: 0.001200500157661736, val loss: 0.001399349314471086, val acc: 0.87
# Epoch 7: Train loss: 0.001158544867299497, val loss: 0.0013236024305224418, val acc: 0.88
# Epoch 8: Train loss: 0.0011060488118479648, val loss: 0.0012889314393202463, val acc: 0.88
# Epoch 9: Train loss: 0.001077000884960095, val loss: 0.0012606025164326033, val acc: 0.88
# Epoch 10: Train loss: 0.0010206997310742735, val loss: 0.001314953077584505, val acc: 0.88
# Epoch 11: Train loss: 0.000995898738813897, val loss: 0.0012597508902351061, val acc: 0.88
# Epoch 12: Train loss: 0.0009687359556555748, val loss: 0.0012413196315368016, val acc: 0.88
# Epoch 13: Train loss: 0.0009348346749320627, val loss: 0.0012596274067958196, val acc: 0.88
# Epoch 14: Train loss: 0.0009094814530884226, val loss: 0.0012128572252889473, val acc: 0.89
# Training took 168.78360557556152 sec
# Test loss: 0.0013343375295400619, test acc: 0.88
