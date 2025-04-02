
"""
This implementation is based on the PointNet architecture introduced in:
'PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation'
by Qi et al. (2017) https://arxiv.org/abs/1612.00593
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(mlp, self).__init__()

        self.batch_norm = batch_norm
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.batch_norm:
            return self.relu(self.bn(self.conv1(x)))
    
        return self.conv1(x)
        
        
class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k
        self.mlp1 = mlp(k, 64)
        self.mlp2 = mlp(64, 512)
        self.fc3 = nn.Linear(512, k*k)

        self.bn2 = nn.BatchNorm1d(512)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.mlp1(x)
        x = self.mlp2(x)
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
     
        x = self.fc3(x)
        
        x = x.view(batch_size, self.k, self.k)
        
        identity = torch.eye(self.k, device=x.device).view(1, self.k, self.k)
        x = x + identity
        
        return x

      

class PointNetSeg(nn.Module):
    def __init__(self):
        super(PointNetSeg, self).__init__()
        self.TNet3 = TNet(3)
        self.TNet64 = TNet(64)

        # Classication Network for global features
        self.mlp1 = mlp(3, 64)
        self.mlp2 = mlp(64, 64)
        self.mlp4 = mlp(64, 256)
        self.mlp5 = mlp(256, 512)
  
        # Segmentation Network
        self.mlp6 = mlp(576, 256)
        self.mlp7 = mlp(256, 4, batch_norm=False)

    def forward(self, x):
        x_ = x.clone()

        T_3 = self.TNet3(x_)
        x = torch.matmul(T_3, x)

        x = self.mlp1(x)
        x = self.mlp2(x)
       
        x_ = x.clone()
        T_64 = self.TNet64(x_)
        x = torch.matmul(T_64, x)

        x_feature = x.clone()

        x = self.mlp4(x)
        x = self.mlp5(x)

        x_global = torch.max(x, 2, keepdim=True)[0]
        x_global = x_global.expand(-1, -1, x_feature.shape[-1])


        x_concat = torch.cat([x_global, x_feature], dim=1)

        x = self.mlp6(x_concat)
        x = self.mlp7(x)

        return x, T_64
