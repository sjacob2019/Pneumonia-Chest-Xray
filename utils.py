import pandas as pd
import matplotlib.pyplot as plt
import os
from cv2 import imread, resize, IMREAD_GRAYSCALE
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay

class ImgDataset(Dataset):
    def __init__(self, df, img_size=(96, 96), transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path, label = self.df.iloc[idx]
        img = imread(img_path, IMREAD_GRAYSCALE)
        if img is None:
            print(f"Didn't read image at {img_path}")
        img = resize(img, self.img_size)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

def get_data():
    paths = {
        'training': './data/chest_xray/chest_xray/train/',
        'validation': './data/chest_xray/chest_xray/val/',
        'test': './data/chest_xray/chest_xray/test/'
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
    dataset = dataset[~dataset['filename'].str.contains('DS_Store')] # Exclude DS_Store
    dataset = dataset.sample(frac=1).reset_index(drop=True) # Shuffle Dataset
    return dataset

def get_dataloader(data_df, batch_size, img_size=(96, 96), transform=ToTensor(), shuffle=True):
    dataset = ImgDataset(data_df, img_size=img_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle)
    return dataloader

def train_loop(dataloader, model, optimizer, device, history, class_weights=None):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    y_true, y_pred = torch.Tensor().to(device), torch.Tensor().to(device)
    avg_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.float().to(device)
        output = model(X).flatten()

        # Compute loss
        loss = bce_loss_logits(output, y, weights=class_weights)
        avg_loss += loss.item()

        # Add to total predictions and ground truths
        output = torch.sigmoid(output)
        y_pred = torch.cat((y_pred, torch.round(output).to(device))).to(device)
        y_true = torch.cat((y_true, y)).to(device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    avg_loss /= num_batches
    history['losses'].append(avg_loss)
    calc_metrics(y_pred.int().cpu().detach().numpy(), y_true.int().cpu().detach().numpy(), history)
    

def evaluate(dataloader, model, device, history, mode='val', class_weights=None):
    model.eval()
    num_batches = len(dataloader)
    y_true, y_pred = torch.Tensor().to(device), torch.Tensor().to(device)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            # Get model outputs
            X, y = X.to(device), y.float().to(device)
            output = model(X).flatten()

            # Compute Loss
            loss = bce_loss_logits(output, y, weights=class_weights)
            test_loss += loss

            # Build up y_true and y_pred
            output = torch.sigmoid(output)
            y_pred = torch.cat((y_pred, torch.round(output))) 
            y_true = torch.cat((y_true, y))

    test_loss /= num_batches
    if mode == 'val':
        history['val_losses'].append(test_loss.item())
    accuracy, precision, recall, specificity = calc_metrics(y_pred.int().cpu().detach().numpy(), y_true.int().cpu().detach().numpy(), history, mode)

    print("Test Metrics:")
    print(f"Loss: {test_loss:>8f}, Accuracy: {(100*accuracy):>0.1f}%, Precision: {(100*precision):>0.1f}%, Recall: {(100*recall):>0.1f}%, Specificity: {(100*specificity):>0.1f}%\n")
    return y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy()

def calc_metrics(y_pred, y_true, history=None, mode='train'):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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

def bce_loss_logits(logits, target, weights=None):
    logits = logits.double()
    target = target.double()
    a = target * F.logsigmoid(logits)
    b = (1 - target) * torch.log(1 - torch.sigmoid(logits))
    if weights is not None:
        loss = -1.0 * (weights[1] * a + weights[0] * b)
    else:
        loss = -1.0 * (a + b)
    loss = loss.mean()
    return loss

def save_model(model, path, history, used_weights=False):
    if not os.path.exists('history'):
        os.mkdir('history')
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    dict_path = './history/weights.json' if used_weights else './history/normal.json'
    with open(dict_path, 'w') as fp:
        json.dump(history, fp)
    return torch.save(model, path)

def plot_confusion_matrix(y_pred, y_true):
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia'])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()
