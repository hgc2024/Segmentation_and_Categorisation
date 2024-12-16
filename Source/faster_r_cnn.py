"""
Results with 5 epochs, batches of 4, and 100 training images and 20 validation images:
Training Epoch 1/5:  92%|███ting: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:33<00:00,  6.79s/it]
Mean IoU: 0.0916, Detection Rate (IoU>=0.5): 0.0483
Epoch 1, Train Loss: 84.9360, Mean IoU: 0.0916, Detection Rate: 0.0483
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:28<00:00,  5.68s/it]
Mean IoU: 0.0871, Detection Rate (IoU>=0.5): 0.0414
Epoch 2, Train Loss: 46.0522, Mean IoU: 0.0871, Detection Rate: 0.0414
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:37<00:00,  7.51s/it]
Mean IoU: 0.2395, Detection Rate (IoU>=0.5): 0.1276
Epoch 3, Train Loss: 44.1899, Mean IoU: 0.2395, Detection Rate: 0.1276
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:35<00:00,  7.02s/it]
Mean IoU: 0.1793, Detection Rate (IoU>=0.5): 0.1310
Epoch 4, Train Loss: 44.1052, Mean IoU: 0.1793, Detection Rate: 0.1310
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:33<00:00,  6.70s/it] 
Mean IoU: 0.1684, Detection Rate (IoU>=0.5): 0.1172
Epoch 5, Train Loss: 44.9708, Mean IoU: 0.1684, Detection Rate: 0.1172

"""


import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import numpy as np
import cv2
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard normalisation parameters for ImageNet pretrained models
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Compose transform to include normalisation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Dataset class for COCO subset
class COCOSubsetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, subset_size=1000):
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

        # Apply transform if available
        if self.transform is not None:
            img_tensor = self.transform(image)

        return img_tensor, target


def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
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
    """
    Evaluate the model using IoU-based metric.
    For each ground-truth box, find the best predicted box.
    Compute the IoU and consider it a correct detection if IoU >= iou_threshold.
    Returns: mean IoU over all matched boxes and the detection rate at the given threshold.
    """
    model.eval()
    all_ious = []
    matches = 0
    total_gt = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            # Only images, no targets for inference
            predictions = model(images)

            # Move targets to CPU for evaluation
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            for pred, target in zip(predictions, targets):
                gt_boxes = target['boxes'].numpy()
                pred_boxes = pred['boxes'].cpu().numpy()
                total_gt += len(gt_boxes)

                # Match each GT box to the predicted box with the highest IoU
                # (This is a simplistic matching strategy)
                for gt_box in gt_boxes:
                    ious = [compute_iou(gt_box, pb) for pb in pred_boxes]
                    if ious:
                        best_iou = max(ious)
                        all_ious.append(best_iou)
                        if best_iou >= iou_threshold:
                            matches += 1
                    else:
                        # No predictions, so IoU = 0 for this GT
                        all_ious.append(0.0)

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    detection_rate = matches / total_gt if total_gt > 0 else 0.0

    print(f"Mean IoU: {mean_iou:.4f}, Detection Rate (IoU>={iou_threshold}): {detection_rate:.4f}")
    return mean_iou, detection_rate


def train_model(model, train_loader, val_loader, num_epochs, optimizer, model_save_path=None):

    best_mean_iou = 0.0

    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model.train()
        epoch_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, targets in train_iterator:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            if isinstance(loss_dict, list):
                # If no annotations are present, skip this batch
                continue

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            train_iterator.set_postfix(loss=losses.item())

        # Use the new evaluation metric after each epoch
        mean_iou, detection_rate = evaluate_model(model, val_loader, iou_threshold=0.5)

        if model_save_path is not None and mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), model_save_path)

        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Mean IoU: {mean_iou:.4f}, Detection Rate: {detection_rate:.4f}")


def main():
    # File paths
    train_image_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Images"
    train_annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
    val_image_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Eval_Images"
    val_annotation_file = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_val2017.json"
    model_save_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Models\best_model_faster_r_cnn_v2.h5"

    model_load_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Models\best_model_faster_r_cnn_v1.h5"

    # Create datasets and dataloaders
    train_dataset = COCOSubsetDataset(train_image_dir, train_annotation_file, transform=transform, subset_size = 8515)
    val_dataset = COCOSubsetDataset(val_image_dir, val_annotation_file, transform=transform, subset_size=355)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize the Mask R-CNN model
    num_classes = 91  # COCO dataset has 90 classes + background

    # use model from model_load_path
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
    model.load_state_dict(torch.load(model_load_path)) # Comment this line out if you want to train from scratch

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train and validate the model
    train_model(model, train_loader, val_loader, num_epochs=40, optimizer=optimizer, model_save_path=model_save_path)

if __name__ == "__main__":
    main()
