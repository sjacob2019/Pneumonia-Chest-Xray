import os
import pandas as pd
import matplotlib.pyplot as plt

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