from turtle import forward
from typing import List
import torch.nn as nn
import torch
import torch.nn.functional as F
class myCNN(nn.Module):
    
    def __init__(self,kernls:List[int]):
        super(myCNN,self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=len(kernls),kernel_size=(i,75))
            for i in kernls
        ])
        self.fc = nn.Linear(len(kernls)**2,out_features=2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, file):
        pass
        