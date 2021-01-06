import numpy as np
import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    
    def MAPE(self, pred, target):
        idx = (target>0.2).nonzero()
        # print(max(torch.abs(target[idx] - pred[idx])))
        # print(max(torch.abs(target[idx] - pred[idx]) / target[idx]))
        return torch.mean(torch.abs(target[idx] - pred[idx]) / target[idx])
    
    def RMSE(self, pred, target):
        return torch.mean((target - pred)**2)**0.5