import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class for COCO subset
class COCOSubsetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, subset_size=100):
        with open(annotation_file) as f:
            self.coco_data = json.load(f)

        available_images = set(os.listdir(image_dir))
        self.images = [img for img in self.coco_data['images'] if img['file_name'] in available_images][:subset_size]
        valid_image_ids = {img['id'] for img in self.images}
        self.annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] in valid_image_ids]

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
            # Create mask from segmentation polygons
            if not isinstance(ann['segmentation'], list):
                continue
            mask = np.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=np.uint8)
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape(-1, 2)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

            # Add bounding box, label, and mask
            bbox = ann['bbox']  # [x, y, width, height]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(ann['category_id'])
            masks.append(torch.from_numpy(mask).bool())

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.stack(masks) if masks else torch.zeros((0, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.bool),
        }

        return img_tensor, target

# Train the model
def train_model(model, train_loader, val_loader, num_epochs, optimizer):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        val_loss = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss}, Validation Loss: {val_loss}")

# Evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss += sum(loss for loss in loss_dict.values()).item()

    return total_loss / len(data_loader)

# Main function
def main():
    # File paths
    train_image_dir = "/path/to/train/images"
    train_annotation_file = "/path/to/train/annotations.json"
    val_image_dir = "/path/to/val/images"
    val_annotation_file = "/path/to/val/annotations.json"
    model_save_path = "/path/to/save/model.pth"

    # Create datasets and dataloaders
    train_dataset = COCOSubsetDataset(train_image_dir, train_annotation_file)
    val_dataset = COCOSubsetDataset(val_image_dir, val_annotation_file, subset_size=20)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize the Mask R-CNN model
    num_classes = 91  # COCO dataset has 90 classes + background
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train and validate the model
    train_model(model, train_loader, val_loader, num_epochs=5, optimizer=optimizer)

    # Save the model
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    main()
