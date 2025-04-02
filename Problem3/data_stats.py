import os
import glob
from tqdm import tqdm

def find_max_lines_in_pts_files(directory_path):
    # Get all .pts files in the specified directory
    pts_files = glob.glob(os.path.join(directory_path, "*.pts"))
    
    if not pts_files:
        print("No .pts files found in the specified directory")
        return 0
    
    max_lines = 0
    max_file = ""
    
    # Iterate through each .pts file with progress bar
    for pts_file in tqdm(pts_files, desc="Processing .pts files"):
        with open(pts_file, 'r') as file:
            # Count the number of lines in the file
            num_lines = sum(1 for line in file)
            
            # Update maximum if current file has more lines
            if num_lines > max_lines:
                max_lines = num_lines
                max_file = pts_file
    
    print(f"Maximum number of lines found: {max_lines}")
    print(f"File with maximum lines: {os.path.basename(max_file)}")
    return max_lines

if __name__ == "__main__":
    # Replace this with your directory path
    directory = "/home/anthony-roumi/Desktop/HW0/Problem3/train/pts"
    find_max_lines_in_pts_files(directory)
