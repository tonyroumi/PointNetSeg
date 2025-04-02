from torch.utils.data import Dataset
import os
import numpy as np
import torch

DATA = '/home/anthony-roumi/Desktop/HW0/Problem3/train/pts'
LABEL = '/home/anthony-roumi/Desktop/HW0/Problem3/train/label'
class PointNetDataset(Dataset):
    def __init__(self, dataset_path, label_path):
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.point_data = sorted(os.listdir(dataset_path))
        self.label_data = sorted(os.listdir(label_path))
        

    def __len__(self):
        return len(self.point_data)
    
    def __getitem__(self, idx):
       
        points = np.loadtxt(os.path.join(DATA, self.point_data[idx]))
        points_tensor = torch.from_numpy(points).float()
         
        label = np.loadtxt(os.path.join(LABEL, self.label_data[idx]))
        label_tensor = torch.from_numpy(label).long()

        
        return points_tensor, label_tensor
        