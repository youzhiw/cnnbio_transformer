#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import torch
import os
import math
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import swin_t, Swin_T_Weights
import torch.optim as optim
from tqdm import tqdm  



device = torch.device("cuda:2")  # Fourth GPU, as the index is 0-based



train_dir = '/GILMLab/GILMLabProjects/DeepLearning/deepquantification/data/GDCAtlas-Data/patches/train' 
test_dir = '/GILMLab/GILMLabProjects/DeepLearning/deepquantification/data/GDCAtlas-Data/patches/test'




transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class GDC(Dataset):
    def __init__(self, src_dir, transforms = None):
        self.src_dir = src_dir
        csv_dir = os.path.join(src_dir, 'metadata.csv')
        df = pd.read_csv(csv_dir)
        self.images = df['file_name'].values
        self.labels = df['label'].values
        self.transforms = transforms
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        image = Image.open(os.path.join(self.src_dir, img))
        label = self.labels[idx]
            
        if self.transforms:
            img_tensor = self.transforms(image)
        else:
            img_tensor = torch.from_numpy(np.float32(image)).unsqueeze(0)
        return img_tensor, label




train_dataset = GDC(train_dir, transform)
train_dataloader = DataLoader(train_dataset, batch_size = 80, shuffle = True)
test_dataset = GDC(test_dir, transform)
test_dataloader = DataLoader(test_dataset, batch_size = 80, shuffle = True)


class SwinT(nn.Module):
    def __init__(self):
        super(SwinT, self).__init__()
        self.model1 = swin_t(weights = Swin_T_Weights.IMAGENET1K_V1)
        
        self.fc1 = nn.Linear(1000, 3)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.model1(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

model = SwinT()

def train_net(net, epochs, train_dataloader, valid_loader, optimizer, loss_function, device ):
    net.to(device)
    ret_train_loss = []
    ret_valid_loss = []

    for epoch in range(epochs):
        net.train()

        train_loss = []
        for i, (img, label) in enumerate(train_dataloader):
            img= img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            y_pred = net(img)
            loss = loss_function(y_pred, label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_epoch_loss = np.mean(train_loss)
        ret_train_loss.append(train_epoch_loss)

        # Validation phase
        net.eval()
        valid_loss = []
        with torch.no_grad():
            for i, (img, label) in enumerate(valid_loader):
                img = img.to(device)
                label = label.to(device)
                y_pred = net(img)
                loss = loss_function(y_pred, label)
                valid_loss.append(loss.item())

        epoch_vloss = np.mean(valid_loss)

        print(f"epoch {epoch} | training loss: {train_epoch_loss:.4f} | validation loss: {epoch_vloss:.4f}")
        ret_valid_loss.append(epoch_vloss)

    return ret_train_loss, ret_valid_loss




optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_function = nn.CrossEntropyLoss()




check_dir = "/home/youzhiwang/cnnbio_transformer/checkpoints_GDC"
min_valid_loss = math.inf
overall_train_loss = []
overall_valid_loss = []

for epoch in tqdm(range(100)):
    train_loss, valid_loss = train_net(model, 1, train_dataloader, test_dataloader, optimizer, loss_function, device)
    if valid_loss[0] < min_valid_loss:
        min_valid_loss = valid_loss[0]
        model_filename = f'Epoch_{epoch}_VLoss_{valid_loss[0]:.4f}.pth'
        model_path = os.path.join(check_dir, model_filename)
        torch.save(model.state_dict(), model_path)
    overall_train_loss.append(train_loss)
    overall_valid_loss.append(valid_loss)
        

np.savetxt('training_loss.txt', overall_train_loss)

np.savetxt('valid_loss.txt', overall_valid_loss)

