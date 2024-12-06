"""
Please adjust the code later, as this is merely a placeholder template.

Key Areas to Fill In

    Bounding Box Data in preprocess_image:
        Replace torch.tensor([[15, 20, 120, 200]]) with the actual bounding box data for your image.

    Labels in preprocess_image:
        Replace torch.tensor([1]) with the actual class labels corresponding to the bounding boxes.

    Segmentation Masks in preprocess_image:
        Ensure that the segmentation path and logic generate meaningful binary masks for each object category.

    Backbone Output Channels:
        Verify that the final output channels (256) of CustomBackbone match the expectation of the MaskRCNN model.

    RoI Align Layers:
        If needed, replace torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2) with your own implementation for Region of Interest (RoI) pooling.

    Dataset-Specific Paths:
        Update the paths for image_path and segmented_path to point to your actual dataset.

    Number of Classes (num_classes):
        Update this based on your dataset's number of categories (including background).

    Custom Training and Loss Functions (Optional):
        Customize the forward method or integrate specific loss functions if needed.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Define a simple custom Backbone with Spectral Coordinate Block
class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()
        self.out_channels = 256  # Add this line to specify output channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.sc_block1 = SpectralCoordinateBlock(64, 128)
        self.sc_block2 = SpectralCoordinateBlock(128, self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.sc_block1(x)
        x = self.sc_block2(x)
        return x

# Define Spectral Coordinate Block manually
class SpectralCoordinateBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpectralCoordinateBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.rope_embedding = nn.Parameter(torch.randn(out_channels, 1, 1))  # Rotary Position Embedding

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x * self.rope_embedding  # Apply Rotary Position Embedding
        return x

# Integrate Custom Backbone with Mask R-CNN
class CustomMaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomMaskRCNN, self).__init__()
        self.backbone = CustomBackbone()
        self.out_channels = 256  # This matches the final output channels of the backbone

        # RPN setup
        self.rpn_anchor_gen = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # Mask R-CNN components
        # Replace with custom RoI Align layers if needed
        self.model = MaskRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=self.rpn_anchor_gen,
            box_roi_pool=torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)
        )

    def forward(self, images, targets=None):
        # Customize this function for additional pre/post-processing of the inputs/outputs
        return self.model(images, targets)

# Custom Dataset for COCO
class COCOSubsetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, subset_size=100):
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform
        self.subset_size = subset_size
        self.images = self.coco_data['images'][:self.subset_size]
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']

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

# Visualizing the results
def visualize_results(image_input, outputs, threshold=0.05):  # Lower threshold
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = (image_input * 255).astype(np.uint8)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    if len(outputs['boxes']) == 0:
        print("No detections found in the image")
    else:
        print(f"Found {len(outputs['boxes'])} potential objects")
        print(f"Max confidence score: {outputs['scores'].max().item():.3f}")
        
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
            ax.text(x1, y1, f"Label {label.item()}: {score.item():.2f}", color='white', fontsize=8, bbox=dict(facecolor='red', edgecolor='none', alpha=0.5))

    plt.axis("off")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Paths
    annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
    image_dir = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\train2017\train2017'

    # Load subset of COCO dataset
    subset_size = 100
    dataset = COCOSubsetDataset(image_dir, annotation_file, subset_size=subset_size)
    def collate_fn(batch):
        return tuple(zip(*batch))
        
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)  # Set num_workers to 0 to avoid multiprocessing issues

    # Initialize model
    num_classes = 91  # Update based on the number of classes in COCO dataset (including background)
    model = CustomMaskRCNN(num_classes=num_classes)
    model.train()

    # Define optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    

# Training loop (1 epoch for dry run)
    for epoch in range(5):
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}")
        running_loss = 0.0
        for i, (images, targets) in progress_bar:
            # Skip batches with empty boxes
            if any(len(target['boxes']) == 0 for target in targets):
                continue

            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                progress_bar.set_postfix(loss=losses.item())
            except Exception as e:
                print(f"Error in batch {i}: {str(e)}")
                continue

    # Print training complete message
    print("Training complete.")

    # Switch to evaluation mode
    model.eval()

    # Evaluate on a small set of images
    eval_subset_size = 20
    eval_threshold = 0.5
    eval_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\val2017\val2017"
    annotation_file = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_val2017.json"
    eval_dataset = COCOSubsetDataset(eval_dir, annotation_file, subset_size=eval_subset_size)
    eval_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))  # Set num_workers to 0 to avoid multiprocessing issues

    with torch.no_grad():
        for i, (images, targets) in enumerate(eval_data_loader): # Something is wrong when loading images, as 000000391895.jpg isn't in the test2017 folder
            images = list(image for image in images)
            outputs = model(images)
            print(f"Evaluating image {i + 1}/{eval_subset_size}")
            # print the bounding boxes
            print(outputs[0]['boxes'])
            visualize_results(images[0].cpu().numpy().transpose(1, 2, 0), outputs[0], threshold=0.05)
            # Calculate metrics for this image
            gt_boxes = targets[0]['boxes']
            pred_boxes = outputs[0]['boxes']
            pred_scores = outputs[0]['scores']
            
            # Calculate IoU between predicted and ground truth boxes
            ious = torch.zeros(len(pred_boxes), len(gt_boxes))
            for p_idx, pred_box in enumerate(pred_boxes):
                for g_idx, gt_box in enumerate(gt_boxes):
                    # Calculate intersection coordinates
                    x1 = max(pred_box[0], gt_box[0])
                    y1 = max(pred_box[1], gt_box[1])
                    x2 = min(pred_box[2], gt_box[2])
                    y2 = min(pred_box[3], gt_box[3])
                    
                    # Calculate intersection area
                    intersection = max(0, x2 - x1) * max(0, y2 - y1)
                    
                    # Calculate union area
                    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union = pred_area + gt_area - intersection
                    
                    ious[p_idx, g_idx] = intersection / union

            # Calculate average precision
            matched_gt = set()
            true_positives = 0
            for pred_idx in torch.argsort(pred_scores, descending=True):
                if torch.max(ious[pred_idx]) > 0.5:  # IoU threshold of 0.5
                    best_gt_idx = torch.argmax(ious[pred_idx]).item()
                    if best_gt_idx not in matched_gt:
                        true_positives += 1
                        matched_gt.add(best_gt_idx)

            precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0
            recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1_score:.3f}")

    # Print evaluation complete message
    print("Evaluation complete.")


