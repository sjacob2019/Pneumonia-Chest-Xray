import pandas as pd
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from utils import *

class ImgDataset(Dataset):
    def __init__(self, df, img_size=(28, 28), transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path, label = self.df.iloc[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

def get_data():
    paths = {
        'training': './data/chest_xray/train/',
        'validation': './data/chest_xray/val/',
        'test': './data/chest_xray/test/'
    }
    categories = {0: 'normal', 1: 'pneumonia'}
    dataset = pd.DataFrame()
    for cat_number, cat_name in categories.items():
        for path in paths.values():
            combined_path = os.path.join(path, cat_name.upper())
            files = os.listdir(combined_path)
            for i, file in enumerate(files):
                files[i] = os.path.join(combined_path, file)
            tmp = pd.DataFrame()
            tmp['filename'] = files
            tmp['class'] = cat_number
            dataset = pd.concat([dataset, tmp])
    dataset = dataset.sample(frac=1).reset_index(drop=True) # Shuffle Dataset
    return dataset

def get_dataloader(data_df, batch_size, transform=ToTensor(), shuffle=True):
    dataset = ImgDataset(data_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle)
    return dataloader


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.float().to(device)
        pred = model(X).flatten()
        correct += (torch.round(pred) == y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    

def evaluate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.float().to(device)
            pred = model(X).flatten()
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def save_model(model, path):
    return torch.save(model, path)