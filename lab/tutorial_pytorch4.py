# fashion mnist

# sources:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

import numpy as np
import time
import math
import torch
from torchvision import datasets, transforms
from torch import nn

# use gup if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float


BATCH_SIZE=64
LEARNING_RATE=1e-2
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
        model.zero_grad() # Zero the gradients before running the backward pass
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= LEARNING_RATE * param.grad


    val_loss = 0
    val_success = 0
    for images, labels in valloader:
        images = images.to(device)
        labels = labels.to(device)
        log_probs = model(images)

        loss = loss_fun(log_probs, labels) # negative log likelihood loss
        val_loss += loss.item()
        val_success += success_count(log_probs, labels).item()

    print(f'Train loss: {train_loss/train_size}, val loss: {val_loss/val_size}, val acc: {val_success/val_size}')

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

print(f'Test loss: {test_loss/test_size}, test acc: {test_success/test_size}')


# Sequential(
#   (0): Flatten(start_dim=1, end_dim=-1)
#   (1): Linear(in_features=784, out_features=128, bias=True)
#   (2): ReLU()
#   (3): Linear(in_features=128, out_features=64, bias=True)
#   (4): ReLU()
#   (5): Linear(in_features=64, out_features=10, bias=True)
#   (6): LogSoftmax(dim=1)
# )
# Train loss: 0.017345405626421173, val loss: 0.010274952851235867, val acc: 0.765
# Train loss: 0.009097970558951299, val loss: 0.008513558454811573, val acc: 0.8050833333333334
# Train loss: 0.007983076673311492, val loss: 0.007830930712322394, val acc: 0.8186666666666667
# Train loss: 0.007418800723428528, val loss: 0.007437275769809882, val acc: 0.8309166666666666
# Train loss: 0.007028691137209535, val loss: 0.007013617708037297, val acc: 0.84025
# Train loss: 0.006729549628371994, val loss: 0.006799692949901025, val acc: 0.84375
# Train loss: 0.006488321645495792, val loss: 0.006543361307432254, val acc: 0.8511666666666666
# Train loss: 0.006297220619395375, val loss: 0.006434765993307034, val acc: 0.8553333333333333
# Train loss: 0.00612216599813352, val loss: 0.006271257766832908, val acc: 0.8584166666666667
# Train loss: 0.0059730245002234976, val loss: 0.006206350930035114, val acc: 0.8584166666666667
# Train loss: 0.005860088921772937, val loss: 0.00605774100124836, val acc: 0.8609166666666667
# Train loss: 0.0057268819079423945, val loss: 0.005986617520451546, val acc: 0.864
# Train loss: 0.005616945830173791, val loss: 0.005856479530533155, val acc: 0.86675
# Train loss: 0.0055046686138957735, val loss: 0.005895494812478622, val acc: 0.8644166666666667
# Train loss: 0.005407761647365988, val loss: 0.005763836787392696, val acc: 0.8670833333333333
# Training took 224.89285707473755 sec
# Test loss: 0.006233267250657082, test acc: 0.8598
