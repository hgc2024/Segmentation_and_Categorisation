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

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
        # Ensure images and targets are on the same device
        images = [img.to(device) for img in images]
        if targets is not None:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
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
def visualize_results(image_input, outputs, threshold=0.5, image_id = None):  # Lower threshold
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
        # Load COCO categories
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        categories = coco_data['categories']

        print(f"Found {len(outputs['boxes'])} potential objects")
        print(f"Max confidence score: {outputs['scores'].max().item():.3f}")

        for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
            if score > threshold:
                x1, y1, x2, y2 = box
                category_name = next((cat['name'] for cat in categories if cat['id'] == label.item()), "Unknown")
                ax.add_patch(plt.Rectangle((x1.cpu(), y1.cpu()), (x2 - x1).cpu(), (y2 - y1).cpu(), fill=False, color='red', linewidth=2))
                ax.text(x1.cpu(), y1.cpu(), f"{category_name}: {score.item():.2f}", color='white', fontsize=8, bbox=dict(facecolor='red', edgecolor='none', alpha=0.5))
        print(f"Found {len(outputs['boxes'])} potential objects")
        print(f"Max confidence score: {outputs['scores'].max().item():.3f}")
        
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1.cpu(), y1.cpu()), (x2 - x1).cpu(), (y2 - y1).cpu(), fill=False, color='red', linewidth=2))
            ax.text(x1.cpu(), y1.cpu(), f"Label {label.item()}: {score.item():.2f}", color='white', fontsize=8, bbox=dict(facecolor='red', edgecolor='none', alpha=0.5))

    plt.axis("off")

    # Save the figure with image_id in filename
    output_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Test_Results\Images"
    if image_id is not None:
        plt.savefig(os.path.join(output_dir, f"{image_id}_output_.png"))
    else:
        plt.savefig(os.path.join(output_dir, "output.png"))

    plt.show()

# Example usage
if __name__ == "__main__":
    # Paths
    annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
    # image_dir = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\train2017\train2017'
    image_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Images"

    # Load subset of COCO dataset
    subset_size = 500
    dataset = COCOSubsetDataset(image_dir, annotation_file, subset_size=subset_size)
    def collate_fn(batch):
        return tuple(zip(*batch))
        
    data_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)  # Set num_workers to 0 to avoid multiprocessing issues

    # Initialize model
    num_classes = 91  # Update based on the number of classes in COCO dataset (including background)
    model = CustomMaskRCNN(num_classes=num_classes).to(device)
    model.train()

    # Define optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Evaluate on a small set of images
    eval_subset_size = 100
    # eval_threshold = 0.5
    # eval_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\val2017\val2017"
    eval_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Eval_Images"
    annotation_file = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_val2017.json"
    eval_dataset = COCOSubsetDataset(eval_dir, annotation_file, subset_size=eval_subset_size)
    eval_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))  # Set num_workers to 0 to avoid multiprocessing issues

    best_loss = float('inf')
    model_save_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Models\best_model.h5"

    for epoch in range(20):
        # Training loop (as before)
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}")
        running_loss = 0.0
        for i, (images, targets) in progress_bar:
            # Skip batches with empty boxes
            if any(len(target['boxes']) == 0 for target in targets):
                continue

            images = list(image for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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

        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(data_loader)
        print(f"\nEpoch {epoch + 1} average loss: {epoch_loss}")

        # Evaluation phase
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for i, (images, targets) in enumerate(eval_data_loader):
                images = list(image for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images, targets)
                if isinstance(outputs, dict):
                    # During training, model returns a dict of losses
                    batch_loss = sum(outputs.values())
                elif isinstance(outputs, list):
                    # During evaluation, model returns a list of predictions
                    continue  # Skip loss calculation during evaluation
                else:
                    # Handle unexpected output type
                    print(f"Unexpected output type: {type(outputs)}")
                    continue
                eval_loss += batch_loss.item()
                
                # Visualize last batch results
                if i == len(eval_data_loader) - 1:
                    outputs = model(images)
                    visualize_results(images[0].cpu().numpy().transpose(1, 2, 0), outputs[0], image_id=f"epoch_{epoch+1}")

        avg_eval_loss = eval_loss / len(eval_data_loader)
        print(f"Evaluation loss: {avg_eval_loss}")

        # Save model if it improves
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with loss: {best_loss}")
        
        model.train()

    print("Training and evaluation complete.")
    print(f"Best model saved with loss: {best_loss}")