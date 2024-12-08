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
import pandas as pd

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
        
        # Get list of available images in directory
        available_images = set(os.listdir(image_dir))
        
        # Filter images to only those that exist in directory
        self.images = [img for img in self.coco_data['images'] 
                      if img['file_name'] in available_images][:self.subset_size]
        
        # Get valid image IDs
        valid_image_ids = {img['id'] for img in self.images}
        
        # Filter annotations to only those that correspond to available images
        self.annotations = [ann for ann in self.coco_data['annotations'] 
                          if ann['image_id'] in valid_image_ids]
        
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
    plt.ioff()  # Turn off interactive mode
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = (image_input * 255).astype(np.uint8)
    fig = plt.figure(figsize=(12, 8))
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

    plt.axis("off")

    # Save the figure with image_id in filename
    output_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Test_Images\v2"
    if image_id is not None:
        plt.savefig(os.path.join(output_dir, f"{image_id}_output_.png"))
    else:
        plt.savefig(os.path.join(output_dir, "output.png"))
    
    plt.close(fig)  # Close the figure to free memory

# Example usage
if __name__ == "__main__":
    # Paths
    annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
    # image_dir = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\train2017\train2017'
    image_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Images"

    # Load subset of COCO dataset
    subset_size = 100
    dataset = COCOSubsetDataset(image_dir, annotation_file, subset_size=subset_size)
    def collate_fn(batch):
        return tuple(zip(*batch))
        
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)  # Set num_workers to 0 to avoid multiprocessing issues

     # Evaluate on a small set of images
    eval_subset_size = 10
    # eval_threshold = 0.5
    # eval_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\val2017\val2017"
    # eval_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Eval_Images"
    # Set total number of images to use
    train_ratio = 0.8  # 80% for training
    
    # Set image directory
    image_base_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Images"
    
    # Calculate train and test sizes
    train_size = int(subset_size * train_ratio)
    test_size = subset_size - train_size
    
    # Create datasets with split
    all_dataset = COCOSubsetDataset(image_base_dir, annotation_file, subset_size=subset_size)
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    
        
    # Update eval_dataset to use test_dataset
    eval_dataset = test_dataset
    eval_dir = image_base_dir  # Keep reference to original directory
    annotation_file = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_val2017.json"
    eval_dataset = COCOSubsetDataset(eval_dir, annotation_file, subset_size=eval_subset_size)
    eval_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))  # Set num_workers to 0 to avoid multiprocessing issues

    best_loss = float('inf')
    model_save_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Models\best_model.h5"



    # Initialize model
    num_classes = 91  # Update based on the number of classes in COCO dataset (including background)
    model = CustomMaskRCNN(num_classes=num_classes).to(device)
    model.train()

    # Define optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Lists to store metrics
    train_losses = []
    test_losses = []
    eval_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': []}


    for epoch in range(2):
        # Clear CUDA cache before each epoch
        torch.cuda.empty_cache()
        
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1} Training")
        
        for i, (images, targets) in progress_bar:
            try:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                else:
                    raise e

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values() if loss is not None and torch.isfinite(loss))
                else:
                    losses = loss_dict  # If loss_dict is already a tensor

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                progress_bar.set_postfix(loss=losses.item())
            except Exception as e:
                print(f"Error in batch {i}: {str(e)}")
                continue

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f"\nEpoch {epoch + 1} training loss: {epoch_train_loss}")

        # Testing phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, targets in test_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)

                # Loss needs to be fixed, use print statements to debug
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values() if loss is not None and torch.isfinite(loss))
                    test_loss += losses.item()
                else:
                    losses = loss_dict
                    print(f"Loss: {losses}")
                    test_loss += losses

        epoch_test_loss = test_loss / len(test_loader)
        test_losses.append(epoch_test_loss)
        print(f"Epoch {epoch + 1} test loss: {epoch_test_loss}")

        # Save model if test loss improves
        if epoch == 0 or epoch_test_loss < min(test_losses[:-1]):
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with test loss: {epoch_test_loss}")

    # Final evaluation
    model.eval()
    eval_results = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(eval_loader):
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            # Calculate metrics for each image
            for output, target in zip(outputs, targets):
                metrics = {
                    'image_id': i,
                    'loss': sum(model(images, [target]).values()).item(),
                    'num_detections': len(output['boxes']),
                    'max_score': output['scores'].max().item() if len(output['scores']) > 0 else 0
                }
                eval_results.append(metrics)
                
                # Visualize results
                visualize_results(images[0].cpu().numpy().transpose(1, 2, 0), output, image_id=f"final_eval_{i}")

    # Save evaluation results
    results_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Test_Results\results_statistics"
    results_file = os.path.join(results_dir, "evaluation_results.csv")
    df = pd.DataFrame(eval_results)
    df.to_csv(results_file, index=False)
    print("Evaluation results saved to evaluation_results.csv")

