# add batchnorm

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
CONV_CHANNELS = [10, 50, 100]
HIDDEN_SIZES = [128]
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
    nn.Conv2d(1, CONV_CHANNELS[0], (3,3)),
    nn.BatchNorm2d(CONV_CHANNELS[0]),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    nn.Conv2d(CONV_CHANNELS[0], CONV_CHANNELS[1], (4,4)),
    nn.BatchNorm2d(CONV_CHANNELS[1]),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    nn.Conv2d(CONV_CHANNELS[1], CONV_CHANNELS[2], (3,3)),
    nn.BatchNorm2d(CONV_CHANNELS[2]),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(CONV_CHANNELS[2] * 3 * 3, HIDDEN_SIZES[0]),
    nn.BatchNorm1d(HIDDEN_SIZES[0]),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZES[0], OUTPUT_SIZE),
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
#   (0): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
#   (1): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU()
#   (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
#   (4): Conv2d(10, 50, kernel_size=(4, 4), stride=(1, 1))
#   (5): BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (6): ReLU()
#   (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
#   (8): Conv2d(50, 100, kernel_size=(3, 3), stride=(1, 1))
#   (9): BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (10): ReLU()
#   (11): Flatten(start_dim=1, end_dim=-1)
#   (12): Linear(in_features=900, out_features=128, bias=True)
#   (13): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (14): ReLU()
#   (15): Linear(in_features=128, out_features=10, bias=True)
#   (16): LogSoftmax(dim=1)
# )
# Epoch 0: Train loss: 0.0018519266011814276, val loss: 0.0012318684831261635, val acc: 0.89
# Epoch 1: Train loss: 0.001083450403685371, val loss: 0.0010873689043025176, val acc: 0.90
# Epoch 2: Train loss: 0.0008977757270137469, val loss: 0.0010354503355920314, val acc: 0.90
# Epoch 3: Train loss: 0.0007716469687099258, val loss: 0.0010076238922774791, val acc: 0.91
# Epoch 4: Train loss: 0.0006633393396623433, val loss: 0.0010120395148793856, val acc: 0.91
# Epoch 5: Train loss: 0.0005704063020336132, val loss: 0.0010440726652741431, val acc: 0.91
# Epoch 6: Train loss: 0.0004921464268118144, val loss: 0.001047170334806045, val acc: 0.91
# Epoch 7: Train loss: 0.0004234201801009476, val loss: 0.001120989630619685, val acc: 0.91
# Epoch 8: Train loss: 0.0003535861761774868, val loss: 0.0011791419064005216, val acc: 0.90
# Epoch 9: Train loss: 0.0003089522796217352, val loss: 0.0011882645959655443, val acc: 0.91
# Epoch 10: Train loss: 0.00027526480339777967, val loss: 0.0012019692833224933, val acc: 0.91
# Epoch 11: Train loss: 0.0002353182218503207, val loss: 0.0013577471139530342, val acc: 0.90
# Epoch 12: Train loss: 0.00018792831496102736, val loss: 0.0013228585198521614, val acc: 0.91
# Epoch 13: Train loss: 0.00016036783890255416, val loss: 0.0014186571290095648, val acc: 0.91
# Epoch 14: Train loss: 0.0001593704967526719, val loss: 0.0014229887773593267, val acc: 0.90
# Training took 177.69444799423218 sec
# Test loss: 0.0015710890382528305, test acc: 0.90
