"""
Phase 7: DL Model Development
Trains PyTorch Multi-Task CNN for Crack Classification & Bounding Box Localization:
1. PyTorch BuildingCrackDataset with normalization and bbox scaling
2. Multi-Task CNN Architecture (Classification Head + Bounding Box Regressor Head)
3. Multi-task Loss (Binary Cross-Entropy + Smooth L1 Loss)
4. Saves trained weights checkpoint to models/crack_detection_model.pth
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class BuildingCrackDataset(Dataset):
    """PyTorch Dataset for Building Crack Detection & Bounding Box Localization."""
    def __init__(self, annotations: list, transform=None, img_size: int = 224):
        self.annotations = annotations
        self.transform = transform
        self.img_size = img_size

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = ann["filepath"]
        
        if not os.path.exists(img_path):
            # Fallback to resolution relative to data directory
            img_path = os.path.join(DATA_DIR, "images", ann["split"], ann["filename"])

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        has_crack = torch.tensor(ann["has_crack"], dtype=torch.float32)
        bbox_norm = torch.tensor(ann["bbox_norm"], dtype=torch.float32)  # [xmin, ymin, xmax, ymax]

        return image_tensor, has_crack, bbox_norm


class MultiTaskCrackDetector(nn.Module):
    """Deep Multi-Task CNN for joint Crack Classification and Bounding Box Localization."""
    def __init__(self):
        super(MultiTaskCrackDetector, self).__init__()

        # Conv Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4 -> 256*4*4 = 4096
        )

        # Classification Head (Crack Present Probability)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Bounding Box Regressor Head (xmin, ymin, xmax, ymax normalized)
        self.bbox_regressor = nn.Sequential(
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
            nn.Sigmoid()  # Bound outputs in range [0, 1]
        )

    def forward(self, x):
        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)

        crack_prob = self.classifier(feat_flat).squeeze(-1)
        bbox_pred = self.bbox_regressor(feat_flat)

        return crack_prob, bbox_pred


def compute_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def train_dl_model(epochs: int = 15, batch_size: int = 32, lr: float = 1e-3):
    """Trains Multi-Task Deep Learning Crack Detector."""
    print("👁️ Starting Phase 7 PyTorch DL Crack Detector Training...")

    ann_path = os.path.join(DATA_DIR, "images", "annotations.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotations not found at {ann_path}. Run Phase 2 first.")

    with open(ann_path, "r") as f:
        annotations = json.load(f)

    train_anns = [a for a in annotations if a["split"] == "train"]
    val_anns = [a for a in annotations if a["split"] == "val"]

    train_dataset = BuildingCrackDataset(train_anns)
    val_dataset = BuildingCrackDataset(val_anns)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Training hardware device: {device}")

    model = MultiTaskCrackDetector().to(device)

    cls_criterion = nn.BCELoss()
    bbox_criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels, bboxes in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            cls_preds, bbox_preds = model(images)

            loss_cls = cls_criterion(cls_preds, labels)
            # Apply bbox loss only on cracked images
            mask = labels > 0.5
            if mask.sum() > 0:
                loss_bbox = bbox_criterion(bbox_preds[mask], bboxes[mask])
            else:
                loss_bbox = torch.tensor(0.0, device=device)

            total_loss = loss_cls + 2.0 * loss_bbox
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * len(labels)
            preds_binary = (cls_preds > 0.5).float()
            train_correct += (preds_binary == labels).sum().item()

        scheduler.step()

        # Validation Loop
        model.eval()
        val_loss, val_correct = 0.0, 0
        total_iou = []

        with torch.no_grad():
            for images, labels, bboxes in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                bboxes = bboxes.to(device)

                cls_preds, bbox_preds = model(images)
                loss_cls = cls_criterion(cls_preds, labels)
                mask = labels > 0.5
                if mask.sum() > 0:
                    loss_bbox = bbox_criterion(bbox_preds[mask], bboxes[mask])
                else:
                    loss_bbox = torch.tensor(0.0, device=device)

                val_loss += (loss_cls + 2.0 * loss_bbox).item() * len(labels)
                preds_binary = (cls_preds > 0.5).float()
                val_correct += (preds_binary == labels).sum().item()

                # Calculate IoU for cracked validation samples
                for b_pred, b_true, lbl in zip(bbox_preds, bboxes, labels):
                    if lbl > 0.5:
                        iou = compute_iou(b_pred.cpu().numpy(), b_true.cpu().numpy())
                        total_iou.append(iou)

        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        mean_iou = np.mean(total_iou) if total_iou else 0.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Mean IoU: {mean_iou:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(MODELS_DIR, "crack_detection_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "mean_iou": mean_iou
            }, checkpoint_path)

    print(f"✅ Saved trained PyTorch crack detector weights to: {os.path.join(MODELS_DIR, 'crack_detection_model.pth')}")
    return model


if __name__ == "__main__":
    train_dl_model(epochs=12)
