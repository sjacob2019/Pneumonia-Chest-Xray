import pandas as pd
import os
from cv2 import imread, resize, IMREAD_GRAYSCALE
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, precision_score

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
        img = imread(img_path, IMREAD_GRAYSCALE)
        img = resize(img, self.img_size)
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

def train_loop(dataloader, model, loss_fn, optimizer, device, history):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    y_true, y_pred = torch.Tensor(), torch.Tensor()
    avg_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.float().to(device)
        output = model(X).flatten()

        # Compute loss
        loss = loss_fn(output, y)
        avg_loss += loss.item()

        # Add to total predictions and ground truths
        y_pred = torch.cat((y_pred, torch.round(output))) 
        y_true = torch.cat((y_true, y))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    avg_loss /= num_batches
    history['losses'].append(avg_loss)
    calc_metrics(y_pred.int(), y_true.int(), history)
    

def evaluate(dataloader, model, loss_fn, device, history, mode='val'):
    num_batches = len(dataloader)
    y_true, y_pred = torch.Tensor(), torch.Tensor()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            # Get model outputs
            X, y = X.to(device), y.float().to(device)
            output = model(X).flatten()

            # Compute Loss
            loss = loss_fn(output, y)
            test_loss += loss

            # Build up y_true and y_pred
            y_pred = torch.cat((y_pred, torch.round(output))) 
            y_true = torch.cat((y_true, y))

    test_loss /= num_batches
    if mode == 'val':
        history['val_losses'].append(test_loss)
    accuracy, precision, recall, specificity = calc_metrics(y_pred.int(), y_true.int(), history, mode)

    print("Test Metrics:")
    print(f"Loss: {test_loss:>8f}, Accuracy: {(100*accuracy):>0.1f}%, Precision: {(100*precision):>0.1f}%, Recall: {(100*recall):>0.1f}%, Specificity: {(100*specificity):>0.1f}%\n")
    return y_pred.numpy(), y_true.numpy()

def calc_metrics(y_pred, y_true, history=None, mode='train'):
    y_true, y_pred = y_true.detach().numpy(), y_pred.detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = balanced_accuracy_score(y_pred, y_true)
    precision = precision_score(y_pred, y_true)
    recall = recall_score(y_pred, y_true).item()
    specificity = tn / (tn + fp)
    if mode == 'train':
        history['accuracies'].append(accuracy)
        history['precisions'].append(precision)
        history['recalls'].append(recall)
        history['specificities'].append(specificity)
    elif mode == 'val':
        history['val_accuracies'].append(accuracy)
        history['val_precisions'].append(precision)
        history['val_recalls'].append(recall)
        history['val_specificities'].append(specificity)
    return accuracy, precision, recall, specificity

def save_model(model, path):
    return torch.save(model, path)
