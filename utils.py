# utils.py
"""
Shared utilities for Assignment 3 scripts.
Contains data loaders, model constructor, evaluation helpers, and denorm.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = "./output"
os.makedirs(OUTDIR, exist_ok=True)

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.247, 0.243, 0.261]

def denorm(x):
    mean = torch.tensor(MEAN).view(1,3,1,1).to(x.device)
    std  = torch.tensor(STD).view(1,3,1,1).to(x.device)
    return torch.clamp(x*std + mean, 0, 1)

def get_loaders(batch_size=128):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32,4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])
    trainset = CIFAR10("./data", train=True, download=True, transform=tf_train)
    testset  = CIFAR10("./data", train=False, download=True, transform=tf_test)
    return trainset, testset, DataLoader(trainset, batch_size, shuffle=True, num_workers=2), DataLoader(testset, batch_size, shuffle=False, num_workers=2)

def create_resnet18(num_classes=10):
    # Use torchvision weights=None to avoid deprecation warning
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def eval_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct/total
