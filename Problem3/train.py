from torch.utils.data import DataLoader, random_split
from dataset import PointNetDataset
from model import PointNetSeg
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as F

LABEL_PATH = '/home/anthony-roumi/Desktop/HW0/Problem3/train/label'
DATASET_PATH =  '/home/anthony-roumi/Desktop/HW0/Problem3/train/pts'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cloud_loss(pred, target, feat_matrix):
    I = torch.eye(feat_matrix.shape[1]).to(device)
    orth = torch.norm(I - torch.matmul(feat_matrix, feat_matrix.transpose(1, 2)))
    loss = F.nll_loss(pred, target.detach(), ignore_index=-1) + 0.001*orth
    return loss

def collate_fn(batch):
    pts, labels = zip(*batch)
    max_pts = max(pt.shape[0] for pt in pts)

    padded_pts = []
    padded_labels = []
    
    for pt, label in zip(pts, labels):
        if pt.shape[0] < max_pts:
            num_duplicates = max_pts - pt.shape[0]
            
        
            duplicate_indices = torch.randint(0, pt.shape[0], (num_duplicates,))
            
            duplicate_indices = torch.clamp(duplicate_indices, 0, pt.shape[0] - 1)
            
            duplicate_pts = pt[duplicate_indices]
            duplicate_labels = label[duplicate_indices]
            
            padded_pt = torch.cat([pt, duplicate_pts], dim=0)
            padded_label = torch.cat([label, duplicate_labels], dim=0)
        else:
            padded_pt = pt
            padded_label = label
            
        padded_pts.append(padded_pt)
        padded_labels.append(padded_label)
    
    padded_pts = torch.stack(padded_pts)
    padded_labels = torch.stack(padded_labels)

    return padded_pts, padded_labels


if __name__ == "__main__":
    full_dataset = PointNetDataset(DATASET_PATH, LABEL_PATH)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(42))  

    train_loader = DataLoader(train_ds, batch_size=16, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)

    model = PointNetSeg(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_epochs = 100
    model = model.to(device)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model = model.train()
        scheduler.step()

        train_loss = 0
        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for batch_idx, (points, target) in enumerate(train_loop):
            points = points.transpose(1,2)
            points, target = points.to(device), target.to(device).long()
            optimizer.zero_grad()

            pred, feat_matrix = model(points)
            pred = pred.view(-1, 4)
            target = target.view(-1)
            loss = cloud_loss(pred, target, feat_matrix)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for points, target in val_loader:
                points, target = points.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
                points = points.permute(0, 2, 1)
                pred, feat_matrix = model(points)
                loss = cloud_loss(pred, target, feat_matrix)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5