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

class PointNetLoss(nn.Module):
    def __init__(self):
        super(PointNetLoss, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss()

    def forward(self, target, pred, feat_matrix):
        orth = torch.norm(torch.eye(feat_matrix.shape[1]).to(device) - torch.matmul(feat_matrix, feat_matrix.transpose(1, 2)))
        loss = self.nll_loss(pred, target-1) + 0.001*orth
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

def compute_iou(preds, labels):
    iou_list = []
    for cls in range(4):
        tp = torch.sum((preds == cls) & (labels == cls)).float()
        fp = torch.sum((preds == cls) & (labels != cls)).float()
        fn = torch.sum((preds != cls) & (labels == cls)).float()
        
        iou = tp / (tp + fp + fn + 1e-6)  
        iou_list.append(iou)
    
    return iou_list

def mean_iou(preds, labels):
    iou_list = compute_iou(preds, labels)
    
    mean_iou = torch.mean(torch.tensor(iou_list))
    
    return mean_iou


if __name__ == "__main__":
    full_dataset = PointNetDataset(DATASET_PATH, LABEL_PATH, augment=True)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(42))  

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = PointNetSeg()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    loss_fn = PointNetLoss()
    num_epochs = 100
    model = model.to(device)

    best_val_loss = float('inf')
    epoch_loop = tqdm(range(num_epochs), desc='Training Progress')
    for epoch in epoch_loop:
        model = model.train()

        train_loss = 0
        for batch_idx, (points, target) in enumerate(train_loader):
            points = points.transpose(1,2)
            points, target = points.to(device), target.to(device).long()
            optimizer.zero_grad()

            pred, feat_matrix = model(points)
            loss = loss_fn(target, pred, feat_matrix)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for points, target in val_loader:
                points, target = points.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
                points = points.permute(0, 2, 1)
                pred, feat_matrix = model(points)
                loss = loss_fn(target, pred, feat_matrix)
                val_loss += loss.item()
                
                pred_classes = torch.argmax(pred, dim=1) + 1  
                
                all_preds.append(pred_classes.flatten())
                all_targets.append(target.flatten())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        correct = (all_preds == all_targets).sum().item()
        total = all_targets.size(0)
        accuracy = correct / total * 100
        iou = mean_iou(all_preds, all_targets)
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        epoch_loop.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{val_loss:.4f}", iou=f"{iou:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # if (epoch + 1) % 20 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.5