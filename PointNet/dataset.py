from torch.utils.data import Dataset
import os
import numpy as np
import torch

DATA = '/home/anthony-roumi/Desktop/HW0/Problem3/train/pts'
LABEL = '/home/anthony-roumi/Desktop/HW0/Problem3/train/label'
class PointNetDataset(Dataset):
    def __init__(self, dataset_path, label_path, augment=False):
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.point_data = sorted(os.listdir(dataset_path))
        self.label_data = sorted(os.listdir(label_path))
        self.augment = augment
        
    def __len__(self):
        return len(self.point_data)
    
    def __getitem__(self, idx):
       
        points = np.loadtxt(os.path.join(self.dataset_path, self.point_data[idx]))
        points_tensor = torch.from_numpy(points).float()
        
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = torch.tensor([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ], dtype=torch.float32)
            
            points_tensor = torch.matmul(points_tensor, rotation_matrix)
            
            # Add Gaussian noise
            noise = torch.randn_like(points_tensor) * 0.02
            points_tensor = points_tensor + noise
         
        label = np.loadtxt(os.path.join(self.label_path, self.label_data[idx]))
        label_tensor = torch.from_numpy(label).long()
        
        return points_tensor, label_tensor
        