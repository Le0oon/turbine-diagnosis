from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os


class MyData(Dataset):
    def __init__(self,turbin_no,is_test=False):
        Y = pd.read_csv("data/train_labels.csv", index_col="file_name")
        path = os.getcwd()+'/data/' + turbin_no + ('_test' if is_test else '')
        X = np.array([
            torch.from_numpy(np.loadtxt(path + '/'+file, delimiter=',', skiprows=1, encoding='utf-8'))
            for file in os.listdir(path)
        ])
        self.X = torch.from_numpy(X)
        self.trg = [Y.loc[file,'ret'] for file in os.listdir(path)]
        
        
    def __getitem__(self, idx):
        assert idx < len(self.trg)
        
        return self.X[idx],self.trg[idx]
    
    def __len__(self):
        
        return len(self.trg)
        
    