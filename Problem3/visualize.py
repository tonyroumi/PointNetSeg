import numpy as np
import open3d as o3d
import os

def visualize_point_cloud(file_path):
    # Load the point cloud data
    # Assuming the data is in numpy format (.npy) or text format (.txt)
    try:
        if file_path.endswith('.npy'):
            points = np.load(file_path)
        else:
            points = np.loadtxt(file_path)
            
        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Assuming first 3 columns are x,y,z
        
        # Add colors (optional) - setting to blue for visibility
        pcd.paint_uniform_color([0.0, 0.0, 1.0])
        
        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])
        
    except Exception as e:
        print(f"Error loading or visualizing file: {e}")

def main():
    # Specify the path to your point cloud file
    # Replace this with your actual file path
    pts_folder = "train/pts"
    
    # Get the first file in the pts folder as an example
    files = os.listdir(pts_folder)
    if files:
        file_path = os.path.join(pts_folder, files[-1])
        print(f"Visualizing: {file_path}")
        visualize_point_cloud(file_path)
    else:
        print("No files found in the specified directory")

if __name__ == "__main__":
    main()
