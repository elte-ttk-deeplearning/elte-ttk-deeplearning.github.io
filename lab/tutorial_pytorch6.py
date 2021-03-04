# fashion mnist with optimizer using a convolutional architecture

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
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    nn.Conv2d(CONV_CHANNELS[0], CONV_CHANNELS[1], (4,4)),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    nn.Conv2d(CONV_CHANNELS[1], CONV_CHANNELS[2], (3,3)),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(CONV_CHANNELS[2] * 3 * 3, HIDDEN_SIZES[0]),
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
#   (1): ReLU()
#   (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
#   (3): Conv2d(10, 50, kernel_size=(4, 4), stride=(1, 1))
#   (4): ReLU()
#   (5): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
#   (6): Conv2d(50, 100, kernel_size=(3, 3), stride=(1, 1))
#   (7): ReLU()
#   (8): Flatten(start_dim=1, end_dim=-1)
#   (9): Linear(in_features=900, out_features=128, bias=True)
#   (10): ReLU()
#   (11): Linear(in_features=128, out_features=10, bias=True)
#   (12): LogSoftmax(dim=1)
# )
# Epoch 0: Train loss: 0.002904830257718762, val loss: 0.001991182173291842, val acc: 0.81
# Epoch 1: Train loss: 0.0017120744300385315, val loss: 0.0015887971669435502, val acc: 0.85
# Epoch 2: Train loss: 0.0014149679249773424, val loss: 0.0013705369705955188, val acc: 0.88
# Epoch 3: Train loss: 0.0012598678944632411, val loss: 0.0013660506755113601, val acc: 0.87
# Epoch 4: Train loss: 0.001146274789236486, val loss: 0.0012052873199184736, val acc: 0.89
# Epoch 5: Train loss: 0.001054399576348563, val loss: 0.0011847229103247325, val acc: 0.89
# Epoch 6: Train loss: 0.0009911635254199306, val loss: 0.001158401488016049, val acc: 0.89
# Epoch 7: Train loss: 0.0009332152782008052, val loss: 0.001117323330293099, val acc: 0.90
# Epoch 8: Train loss: 0.0008854632340371609, val loss: 0.0011185846887528897, val acc: 0.90
# Epoch 9: Train loss: 0.0008441310391450921, val loss: 0.0010384500212967396, val acc: 0.90
# Epoch 10: Train loss: 0.0007836341070942581, val loss: 0.001065439482529958, val acc: 0.90
# Epoch 11: Train loss: 0.0007514799578736225, val loss: 0.0010281791165471076, val acc: 0.91
# Epoch 12: Train loss: 0.0007045943130118152, val loss: 0.001100350355108579, val acc: 0.90
# Epoch 13: Train loss: 0.0006597844427451491, val loss: 0.0010724349580705165, val acc: 0.90
# Epoch 14: Train loss: 0.0006118121386971324, val loss: 0.0011252571704487007, val acc: 0.90
# Training took 170.0061812400818 sec
# Test loss: 0.0011578142860904335, test acc: 0.90
