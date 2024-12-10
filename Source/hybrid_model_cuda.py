import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            SpectralCoordinateBlock(64, 128),
            SpectralCoordinateBlock(128, out_channels)
        )

    def forward(self, x):
        return {"0": self.features(x)}

class SpectralCoordinateBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.rope_embedding = nn.Parameter(torch.randn(out_channels, 1, 1))

    def forward(self, x):
        return self.conv_block(x) * self.rope_embedding
    
class CustomMaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = CustomBackbone()
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)
        )

    def forward(self, images, targets=None):
        images = [img.to(device) for img in images]
        if targets:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Compute losses if targets are provided
            return self.model(images, targets)
        else:
            # Perform inference if no targets are provided
            return self.model(images)

class COCOSubsetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, subset_size=100):
        with open(annotation_file) as f:
            self.coco_data = json.load(f)
        
        available_images = set(os.listdir(image_dir))
        self.images = [img for img in self.coco_data['images'] 
                      if img['file_name'] in available_images][:subset_size]
        
        valid_image_ids = {img['id'] for img in self.images}
        self.annotations = [ann for ann in self.coco_data['annotations'] 
                          if ann['image_id'] in valid_image_ids]
        
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        img_tensor = to_tensor(image)

        # Get annotations for this image
        annotations = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]

        boxes = []
        labels = []
        masks = []
        for ann in annotations:
            # Validate segmentation data
            if not isinstance(ann['segmentation'], list) or not ann['segmentation']:
                continue
            
            # Convert segmentation to polygons
            poly = []
            for segment in ann['segmentation']:
                if isinstance(segment, (list, tuple)) and len(segment) % 2 == 0:
                    poly.append(np.array(segment).reshape((-1, 2)))
            
            if not poly:
                continue  # Skip if no valid polygons

            # Create mask
            mask = np.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=np.uint8)
            for p in poly:
                cv2.fillPoly(mask, [p.astype(np.int32)], 1)

            # Get bounding box coordinates
            bbox = ann['bbox']  # [x, y, width, height]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(ann['category_id'])
            masks.append(torch.from_numpy(mask).bool())

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.stack(masks) if masks else torch.zeros((0, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.bool),
            'area': torch.tensor([ann['area'] for ann in annotations], dtype=torch.float32),
            'iscrowd': torch.tensor([ann['iscrowd'] for ann in annotations], dtype=torch.int64)
        }

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, target
    
def train_model(model, train_loader, test_loader, optimizer, num_epochs, model_save_path):
    best_test_loss = float('inf')
    metrics = {'train_losses': [], 'test_losses': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        test_loss = evaluate_model(model, test_loader)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), model_save_path)

        metrics['train_losses'].append(train_loss / len(train_loader))
        metrics['test_losses'].append(test_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {metrics['train_losses'][-1]}, Test Loss: {metrics['test_losses'][-1]}")

    return metrics

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            try:
                print(f"Processing batch {batch_idx + 1}...")
                loss_dict = model(images, targets)

                # Ensure the model returned a dictionary of losses
                if not isinstance(loss_dict, dict):
                    raise ValueError(f"Unexpected output type from model: {type(loss_dict)}")

                total_loss += sum(loss for loss in loss_dict.values()).item()

            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")
                raise e

    return total_loss / len(data_loader) if len(data_loader) > 0 else 0

# Main execution code remains similar but uses the helper functions above
def main():
    # Set paths
    train_image_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Images"
    train_annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
    test_image_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Eval_Images"
    test_annotation_file = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_val2017.json"
    model_save_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Models\best_model.h5"

    # Create datasets and dataloaders
    train_dataset = COCOSubsetDataset(train_image_dir, train_annotation_file)
    test_dataset = COCOSubsetDataset(test_image_dir, test_annotation_file, subset_size=20)
    
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize model and optimizer
    num_classes = 91  # COCO dataset has 90 classes + background
    model = CustomMaskRCNN(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    num_epochs = 10
    metrics = train_model(model, train_loader, test_loader, optimizer, num_epochs, model_save_path)

    # Plot training metrics
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['test_losses'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_metrics.png')
    plt.close()

if __name__ == '__main__':
    main()
