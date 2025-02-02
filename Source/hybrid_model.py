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
import csv
from PIL import ImageDraw, ImageFont

# Define a simple custom Backbone with Spectral Coordinate Block
class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()
        self.out_channels = 256
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
        self.rope_embedding = nn.Parameter(torch.randn(out_channels, 1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x * self.rope_embedding
        return x

# Integrate Custom Backbone with Mask R-CNN
class CustomMaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomMaskRCNN, self).__init__()
        self.backbone = CustomBackbone()
        self.out_channels = 256
        self.rpn_anchor_gen = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        self.model = MaskRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=self.rpn_anchor_gen,
            box_roi_pool=torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

# Custom Dataset for COCO
class COCOSubsetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, subset_size=500):
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
        annotations = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]
        boxes = []
        labels = []
        masks = []
        for ann in annotations:
            if not isinstance(ann['segmentation'], list) or not ann['segmentation']:
                continue
            poly = []
            for segment in ann['segmentation']:
                if isinstance(segment, (list, tuple)) and len(segment) % 2 == 0:
                    poly.append(np.array(segment).reshape((-1, 2)))
            if not poly:
                continue
            mask = np.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=np.uint8)
            for p in poly:
                cv2.fillPoly(mask, [p.astype(np.int32)], 1)
            bbox = ann['bbox']
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

# Training loop with evaluation at the end of each epoch
if __name__ == "__main__":
    annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
    image_dir = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\train2017\train2017'
    eval_image_dir = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\val2017\val2017'

    subset_size = 100
    eval_subset_size = 10

    dataset = COCOSubsetDataset(image_dir, annotation_file, subset_size=subset_size)

    # Set the number of CPU threads for PyTorch operations
    num_cpus = 12  # Adjust this number based on your CPU cores
    torch.set_num_threads(num_cpus)
    
    eval_dataset = COCOSubsetDataset(eval_image_dir, annotation_file, subset_size=eval_subset_size)

    data_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    eval_data_loader = DataLoader(eval_dataset, batch_size=5, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = 91
    model = CustomMaskRCNN(num_classes=num_classes)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Prepare CSV file for evaluation results
    csv_file = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\evaluation_results.csv"
    csv_header = ['epoch', 'training_loss', 'evaluation_loss']
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    for epoch in range(1):
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}")
        running_loss = 0.0
        for i, (images, targets) in progress_bar:
            if any(len(target['boxes']) == 0 for target in targets):
                continue
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                import matplotlib.patches as patches

                # Get predictions from model
                predictions = model(images)

                # Create mapping of category IDs to names
                category_id_to_name = {cat['id']: cat['name'] for cat in dataset.categories}

                # Add visualization during evaluation
                def visualize_predictions(image, predictions, save_path=None):
                    # Convert tensor to PIL Image
                    image = torchvision.transforms.ToPILImage()(image)
                    draw = ImageDraw.Draw(image)
                    
                    boxes = predictions['boxes'].cpu()
                    labels = predictions['labels'].cpu()
                    scores = predictions['scores'].cpu()
                    
                    # Draw each prediction
                    for box, label, score in zip(boxes, labels, scores):
                        if score > 0.5:  # Confidence threshold
                            box = box.numpy()
                            label_name = category_id_to_name.get(label.item(), 'Unknown')
                            
                            # Draw rectangle
                            draw.rectangle(box.tolist(), outline='red', width=3)
                            
                            # Draw label
                            draw.text((box[0], box[1]-15), f'{label_name}: {score:.2f}', 
                                     fill='red')
                    
                    if save_path:
                        image.save(save_path)
                    
                    return image

                # In evaluation loop, after model prediction:
                eval_output_dir = 'eval_visualizations'
                os.makedirs(eval_output_dir, exist_ok=True)

                for idx, (image, pred) in enumerate(zip(images, predictions)):
                    vis_path = os.path.join(eval_output_dir, f'eval_img_{idx}.jpg')
                    visualize_predictions(image, pred, vis_path)
                optimizer.step()
                running_loss += losses.item()
                progress_bar.set_postfix(loss=losses.item())
            except Exception as e:
                print(f"Error in batch {i}: {str(e)}")
                continue
        avg_training_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch + 1} complete. Training loss: {avg_training_loss:.4f}")

        # Evaluation at the end of each epoch
        model.eval()
        with torch.no_grad():
            eval_loss = 0.0
            for i, (images, targets) in enumerate(eval_data_loader):
                images = list(image for image in images)
                targets = [{k: v for k, v in t.items()} for t in targets]
                try:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    eval_loss += losses.item()
                except Exception as e:
                    print(f"Error during evaluation batch {i}: {str(e)}")
                    continue
            avg_eval_loss = eval_loss / len(eval_data_loader)
            print(f"Epoch {epoch + 1} Evaluation loss: {avg_eval_loss:.4f}")

        # Save results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_training_loss:.4f}", f"{avg_eval_loss:.4f}"])

        model.train()

    print("Training and evaluation complete.")


