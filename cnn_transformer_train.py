#!/usr/bin/env python
# coding: utf-8


import numpy as np
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

device = torch.device("cuda:3")  # Fourth GPU, as the index is 0-based


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class cnnbio(Dataset):
    def __init__(self, root_dir, transforms = None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, "Images")
        self.labels_dir = os.path.join(self.root_dir, "label")
        self.samples = self.load_samples()
        self.transforms = transforms

    def load_samples(self):
        samples = [img.split('.')[0] for img in os.listdir(self.img_dir) if img.endswith(".jpg")]
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(os.path.join(self.img_dir, sample + ".jpg"))
        with open(os.path.join(self.labels_dir, sample+'.txt')) as file:
            label = np.float32(file.read().split()[-4:])
            
        if self.transforms:
            img_tensor = self.transforms(image)
        else:
            img_tensor = torch.from_numpy(np.float32(image)).unsqueeze(0)
        return img_tensor, label





test_dir = '/GILMLab/GILMLabProjects/DeepLearning/deepquantification/data/cnnbio/cnnbio_ssd_7_10/cnnbio_ssd_test'
test_dataset = cnnbio(test_dir, transform)
test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = True)
train_dir = '/GILMLab/GILMLabProjects/DeepLearning/deepquantification/data/cnnbio/cnnbio_ssd_7_10/cnnbio_ssd_train'
train_dataset = cnnbio(train_dir, transform)
train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)


class SwinT(nn.Module):
    def __init__(self):
        super(SwinT, self).__init__()
        self.model1 = swin_t(weights = Swin_T_Weights.IMAGENET1K_V1)
        
        self.fc1 = nn.Linear(1000, 4)
    def forward(self, x):
        x = self.model1(x)
        x = self.fc1(x)
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
            # print(f'{i + 1}/{len(train_dataloader)}| current training loss: {train_loss[-1]}', end='\r')

        train_epoch_loss = np.mean(train_loss)
        ret_train_loss.append(train_epoch_loss)
        # print(f'epoch {epoch}| training loss: {train_epoch_loss}', end='\r')

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
                # print(f'{i + 1}/{len(valid_loader)}| current validation loss: {valid_loss[-1]}', end='\r')

        epoch_vloss = np.mean(valid_loss)

        print(f"epoch {epoch} | training loss: {train_epoch_loss:.4f} | validation loss: {epoch_vloss:.4f}")
        ret_valid_loss.append(epoch_vloss)

    return ret_train_loss, ret_valid_loss




optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_function = nn.SmoothL1Loss()




check_dir = "/home/youzhiwang/cnnbio_transformer/checkpoints"
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
