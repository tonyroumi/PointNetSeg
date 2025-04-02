import numpy as np
import open3d as o3d
import os
import torch
import argparse
from model import PointNetSeg

def load_model(model_path):
    """Load the trained model from the specified path."""
    model = PointNetSeg()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def visualize_point_cloud_with_predictions(file_path, model=None):
     
    points = np.loadtxt(file_path)
        
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Assuming first 3 columns are x,y,z
    
    # If model is provided, predict classes and color accordingly
    
    points_tensor = torch.tensor(points[:, :3], dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    
    with torch.no_grad():
        pred_logits, _ = model(points_tensor)
        pred_classes = torch.argmax(pred_logits, dim=1).squeeze().numpy()
    
    color_map = {
        0: [1.0, 0.0, 0.0],  
        1: [0.0, 1.0, 0.0],  
        2: [0.0, 0.0, 1.0],  
        3: [0.5, 0.0, 0.5],  
    }
    
    # Assign colors based on predicted classes
    colors = np.array([color_map.get(cls, [0.5, 0.5, 0.5]) for cls in pred_classes])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    

def main():
    model_path = '/home/anthony-roumi/Desktop/HW0/Problem3/best_model.pth'
    test_dir = '/home/anthony-roumi/Desktop/HW0/Problem3/test'
    
    # Load the trained model
    model = load_model(model_path)
    
    # Process each file in the test directory
    files = os.listdir(test_dir)
    for file in files:
        
        file_path = os.path.join(test_dir, file)
        print(f"Visualizing: {file_path}")
        visualize_point_cloud_with_predictions(file_path, model)

if __name__ == "__main__":
    main()
