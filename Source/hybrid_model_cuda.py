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

# Determine the device on which computations will be performed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------
# Model Architecture Components
# ----------------------------------------------
class SpectralCoordinateBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # A simple residual-like block with two convolutional layers, batch norm, and ReLU.
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # A learnable parameter intended to simulate some form of positional embedding or spectral features.
        self.rope_embedding = nn.Parameter(torch.randn(out_channels, 1, 1))

    def forward(self, x):
        # Pass the input through the convolution block and then multiply by the rope_embedding
        # to incorporate some positional/spectral information.
        return self.conv_block(x) * self.rope_embedding

class CustomBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        # This custom backbone begins with a few standard operations and then uses two SpectralCoordinateBlocks.
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
        # Return a dictionary of feature maps keyed by a string as MaskRCNN expects a dict of feature maps.
        return {"0": self.features(x)}

class CustomMaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Instantiate the custom backbone.
        backbone = CustomBackbone()
        # Create the anchor generator for RPN.
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        # Define the Mask R-CNN model using the custom backbone and the given number of classes.
        # The ROI pooling is also specified.
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)
        )

    def forward(self, images, targets=None):
        # Move images to the correct device.
        images = [img.to(device) for img in images]
        if targets is not None:
            # If targets are provided, also move them to the device.
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Return losses (during training).
            return self.model(images, targets)
        else:
            # If no targets are provided, this is an inference call; return predictions.
            return self.model(images)

# ----------------------------------------------
# Dataset
# ----------------------------------------------
class COCOSubsetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, subset_size=100):
        # Load COCO-style annotations.
        with open(annotation_file) as f:
            self.coco_data = json.load(f)

        # Filter images that are actually present in the given directory and limit the subset size.
        available_images = set(os.listdir(image_dir))
        self.images = [img for img in self.coco_data['images'] 
                       if img['file_name'] in available_images][:subset_size]

        # Collect annotations that correspond to the selected images.
        valid_image_ids = {img['id'] for img in self.images}
        self.annotations = [ann for ann in self.coco_data['annotations'] 
                            if ann['image_id'] in valid_image_ids]

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # Number of images in the subset.
        return len(self.images)

    def __getitem__(self, idx):
        # Load one image and its annotations.
        img_info = self.images[idx]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        img_tensor = to_tensor(image)

        # Get annotations for this particular image.
        annotations = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        # Convert annotations to PyTorch tensors.
        for ann in annotations:
            # Make sure we have valid polygon data.
            if not isinstance(ann['segmentation'], list) or not ann['segmentation']:
                continue
            poly = []
            for segment in ann['segmentation']:
                if isinstance(segment, (list, tuple)) and len(segment) % 2 == 0:
                    poly.append(np.array(segment).reshape((-1, 2)))
            if not poly:
                continue

            # Create binary mask from polygons.
            mask = np.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=np.uint8)
            for p in poly:
                cv2.fillPoly(mask, [p.astype(np.int32)], 1)

            # Bounding box and labels.
            bbox = ann['bbox']  # [x, y, w, h]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(ann['category_id'])
            masks.append(torch.from_numpy(mask).bool())
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.stack(masks) if masks else torch.zeros(
                (0, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.bool
            ),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.int64)
        }

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, target

# ----------------------------------------------
# Evaluation Helper Functions
# ----------------------------------------------
def compute_iou(box1, box2):
    # Compute the Intersection-over-Union of two bounding boxes.
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def evaluate_model(model, data_loader, iou_threshold=0.5):
    # Evaluate the model in inference mode using IoU.
    model.eval()
    all_ious = []
    matches = 0
    total_gt = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # No targets passed to model for inference; predictions only.
            predictions = model(images)
            # Move targets to CPU for evaluation.
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            for pred, target in zip(predictions, targets):
                gt_boxes = target['boxes'].numpy()
                pred_boxes = pred['boxes'].cpu().numpy()
                total_gt += len(gt_boxes)

                # For each ground-truth box, find the predicted box with max IoU.
                for gt_box in gt_boxes:
                    ious = [compute_iou(gt_box, pb) for pb in pred_boxes]
                    if ious:
                        best_iou = max(ious)
                        all_ious.append(best_iou)
                        if best_iou >= iou_threshold:
                            matches += 1
                    else:
                        # No predictions means 0 IoU for this GT.
                        all_ious.append(0.0)

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    detection_rate = matches / total_gt if total_gt > 0 else 0.0
    print(f"Mean IoU: {mean_iou:.4f}, Detection Rate (IoU>={iou_threshold}): {detection_rate:.4f}")
    return mean_iou, detection_rate

def train_model(model, train_loader, test_loader, optimizer, num_epochs, model_save_path):
    # Training function that runs multiple epochs.
    best_mean_iou = 0.0
    metrics = {'train_losses': [], 'mean_ious': [], 'detection_rates': []}

    for epoch in range(num_epochs):
        # Clear the GPU cache at the start of each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        model.train()
        epoch_loss = 0.0
        # Loop over the training data batches.
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            # Compute training losses by passing images and targets.
            loss_dict = model(images, targets)
            if isinstance(loss_dict, list):
                # If no annotations, skip this batch.
                continue

            # Sum all losses.
            loss = sum(loss for loss in loss_dict.values())
            # Backpropagate.
            loss.backward()
            optimizer.step()
            # Accumulate training loss.
            epoch_loss += loss.item()

        # After one epoch of training, we evaluate the model on the test set using IoU metrics.
        mean_iou, detection_rate = evaluate_model(model, test_loader, iou_threshold=0.5)

        # If we have improved mean IoU, save the model weights.
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), model_save_path)

        # Store metrics for plotting or analysis later.
        metrics['train_losses'].append(epoch_loss / len(train_loader))
        metrics['mean_ious'].append(mean_iou)
        metrics['detection_rates'].append(detection_rate)

        # Print the epoch summary.
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {metrics['train_losses'][-1]:.4f}, Mean IoU: {mean_iou:.4f}, Detection Rate: {detection_rate:.4f}")

    return metrics

def main():
    # Paths to training and testing images and annotations.
    train_image_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Images"
    train_annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
    test_image_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Eval_Images"
    test_annotation_file = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_val2017.json"
    model_save_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Models\best_model.h5"

    # Create dataset instances.
    train_dataset = COCOSubsetDataset(train_image_dir, train_annotation_file)
    test_dataset = COCOSubsetDataset(test_image_dir, test_annotation_file, subset_size=20)
    
    # Create dataloaders for batching data during training and testing.
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize the custom model with the specified number of classes.
    num_classes = 91  # COCO dataset classes + background.
    model = CustomMaskRCNN(num_classes).to(device)

    # Set up an optimizer for training (Adam).
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run the training loop for the specified number of epochs.
    num_epochs = 10
    metrics = train_model(model, train_loader, test_loader, optimizer, num_epochs, model_save_path)

    # Plot the training loss over epochs.
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.savefig('training_metrics.png')
    plt.close()

    # Plot mean IoU over epochs.
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['mean_ious'], label='Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.legend()
    plt.savefig('mean_iou_metrics.png')
    plt.close()

if __name__ == '__main__':
    main()
