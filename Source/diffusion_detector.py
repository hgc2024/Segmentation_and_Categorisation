import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torchvision.ops import nms, box_iou, generalized_box_iou_loss
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import numpy as np
import cv2
import math
import csv
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. Dataset & Data Handling
# ==========================================

# Standard normalisation parameters for ImageNet pretrained models
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class COCOSubsetDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, subset_size=None):
        with open(annotation_file) as f:
            self.coco_data = json.load(f)

        available_images = set(os.listdir(image_dir))
        # Filter images that are actually in the directory
        self.images = [img for img in self.coco_data['images'] if img['file_name'] in available_images]
        
        if subset_size:
            self.images = self.images[:subset_size]
            
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
        
        # Get annotations for this image
        annotations = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]

        boxes = []
        for ann in annotations:
            # Add bounding box [x, y, w, h] (COCO format is [x, y, w, h])
            bbox = ann['bbox']
            w, h = image.size
            
            # Normalize to [x1, y1, x2, y2] in 0-1 range
            x, y, bw, bh = bbox
            x1 = x / w
            y1 = y / h
            x2 = (x + bw) / w
            y2 = (y + bh) / h
            
            boxes.append([x1, y1, x2, y2])

        if self.transform is not None:
            img_tensor = self.transform(image)
        else:
            img_tensor = to_tensor(image)

        # Convert boxes to tensor
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            # Clamp to ensure valid range
            boxes = torch.clamp(boxes, 0.0, 1.0)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        return img_tensor, boxes

def collate_fn_pad(batch):
    """
    Pads batches of boxes to the maximum number of boxes in the batch (or a fixed number).
    Returns:
        images: [B, C, H, W]
        padded_boxes: [B, Max_Boxes, 4]
        masks: [B, Max_Boxes] (1 for real box, 0 for padding)
    """
    images, boxes_list = zip(*batch)
    images = torch.stack(images, 0)
    
    # Find max boxes in this batch (or set a fixed max like 50)
    max_boxes = 50 
    
    padded_boxes = torch.zeros((len(boxes_list), max_boxes, 4))
    masks = torch.zeros((len(boxes_list), max_boxes))
    
    for i, boxes in enumerate(boxes_list):
        num_boxes = min(boxes.shape[0], max_boxes)
        if num_boxes > 0:
            padded_boxes[i, :num_boxes] = boxes[:num_boxes]
            masks[i, :num_boxes] = 1.0
            
    return images, padded_boxes, masks

# ==========================================
# 2. Diffusion Noise Scheduler (Cosine Schedule)
# ==========================================
class NoiseScheduler:
    def __init__(self, num_timesteps=250, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Cosine schedule
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.alphas_cumprod = alphas_cumprod
        self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clamp(self.betas, 0.0001, 0.9999)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_boxes, t):
        # original_boxes: [B, N, 4]
        # t: [B]
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        noise = torch.randn_like(original_boxes)
        noisy_boxes = sqrt_alpha_bar * original_boxes + sqrt_one_minus_alpha_bar * noise
        return noisy_boxes, noise

# ==========================================
# 3. Model Components (Improved Architecture)
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, inputs):
        # inputs: list of feature maps from backbone (C2, C3, C4, C5)
        # Build laterals
        laterals = [conv(x) for x, conv in zip(inputs, self.lateral_convs)]
        
        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] += F.interpolate(laterals[i], size=prev_shape, mode="nearest")
            
        # Build outputs
        outputs = [conv(x) for x, conv in zip(laterals, self.fpn_convs)]
        return outputs

class DiffusionHead(nn.Module):
    def __init__(self, feature_dim=256, box_dim=4, time_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Cross-attention based head
        self.query_embed = nn.Embedding(50, feature_dim) # Learnable queries for boxes
        
        # Transformer decoder layer
        self.transformer_decoder = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(self.transformer_decoder, num_layers=3)
        
        # Box prediction head
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim + box_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, box_dim)
        )
        
        # Objectness head (is this a real object?)
        self.obj_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 1)
        )
        
        self.box_encoder = nn.Linear(box_dim, feature_dim)

    def forward(self, noisy_boxes, image_features, t):
        # noisy_boxes: [B, N, 4]
        # image_features: [B, C, H, W] (flattened to [B, HW, C])
        # t: [B]
        
        B, N, _ = noisy_boxes.shape
        
        # Time embedding
        t_emb = self.time_mlp(t) # [B, 256]
        t_emb = t_emb.unsqueeze(1) # [B, 1, 256]
        
        # Encode boxes
        box_feat = self.box_encoder(noisy_boxes) # [B, N, 256]
        
        # Add time to box features
        query = box_feat + t_emb
        
        # Transformer cross-attention
        # Memory: image features [B, HW, C]
        # Target: query [B, N, C]
        out_feat = self.transformer(query, image_features) # [B, N, 256]
        
        # Predict noise/box update
        # Concatenate original noisy box to help refinement
        box_input = torch.cat([out_feat, noisy_boxes], dim=-1)
        pred_box = self.box_head(box_input)
        
        # Predict objectness
        pred_obj = self.obj_head(out_feat)
        
        return pred_box, pred_obj

class BoxDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Backbone: ResNet50
        print("Loading ResNet50 backbone...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Extract layers for FPN (C3, C4, C5)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Unfreeze backbone layers 3 and 4
        for param in self.layer1.parameters(): param.requires_grad = False
        for param in self.layer2.parameters(): param.requires_grad = False
        for param in self.layer3.parameters(): param.requires_grad = True
        for param in self.layer4.parameters(): param.requires_grad = True
            
        # FPN
        self.fpn = FeaturePyramidNetwork([512, 1024, 2048], 256)
        
        self.head = DiffusionHead(feature_dim=256, box_dim=4)
        self.scheduler = NoiseScheduler(num_timesteps=250)

    def extract_features(self, images):
        c2 = self.layer1(images)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        fpn_feats = self.fpn([c3, c4, c5])
        
        # Flatten and concatenate features for transformer
        # Use the highest resolution map (P3) or a combination
        # For simplicity, we'll just use P5 (smallest) + P4 + P3 flattened
        
        multi_scale_feats = []
        for feat in fpn_feats:
            B, C, H, W = feat.shape
            multi_scale_feats.append(feat.flatten(2).permute(0, 2, 1)) # [B, HW, C]
            
        combined_feats = torch.cat(multi_scale_feats, dim=1) # [B, Total_HW, C]
        return combined_feats

    def forward(self, images, gt_boxes, masks):
        # gt_boxes: [B, N, 4]
        # masks: [B, N]
        
        device = images.device
        batch_size = images.shape[0]
        
        img_features = self.extract_features(images)
        
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=device).long()
        
        noisy_boxes, noise = self.scheduler.add_noise(gt_boxes, t)
        
        pred_noise, pred_obj = self.head(noisy_boxes, img_features, t)
        
        # 1. Noise MSE Loss (only for valid boxes)
        loss_noise = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=2)
        loss_noise = (loss_noise * masks).sum() / (masks.sum() + 1e-6)
        
        # 2. Box IoU Loss (Predict x0 from noise and compute IoU with GT)
        # x0_pred = (xt - sqrt(1-alpha_bar)*noise_pred) / sqrt(alpha_bar)
        alpha_bar = self.scheduler.alphas_cumprod[t].view(-1, 1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        
        pred_x0 = (noisy_boxes - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
        pred_x0 = torch.clamp(pred_x0, 0, 1)
        
        # GIoU Loss
        loss_giou = 0
        valid_count = 0
        for b in range(batch_size):
            valid_mask = masks[b] == 1
            if valid_mask.sum() > 0:
                loss_giou += generalized_box_iou_loss(pred_x0[b][valid_mask], gt_boxes[b][valid_mask], reduction='mean')
                valid_count += 1
        loss_giou = loss_giou / (valid_count + 1e-6)
        
        # 3. Objectness Loss
        # Target is mask (1 for object, 0 for padding)
        loss_obj = F.binary_cross_entropy_with_logits(pred_obj.squeeze(-1), masks)
        
        return loss_noise + loss_giou + loss_obj

    @torch.no_grad()
    def sample(self, images, num_boxes=50):
        device = images.device
        batch_size = images.shape[0]
        
        img_features = self.extract_features(images)
        
        # Start with random noise
        x = torch.randn((batch_size, num_boxes, 4), device=device)
        
        for i in tqdm(reversed(range(0, self.scheduler.num_timesteps)), desc="Sampling", leave=False):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            pred_noise, pred_obj = self.head(x, img_features, t)
            
            alpha = self.scheduler.alphas[i]
            alpha_cumprod = self.scheduler.alphas_cumprod[i]
            beta = self.scheduler.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * pred_noise) + torch.sqrt(beta) * noise
            
        x = torch.clamp(x, 0, 1)
        
        # Get final objectness scores
        _, pred_obj = self.head(x, img_features, torch.zeros((batch_size,), device=device, dtype=torch.long))
        scores = torch.sigmoid(pred_obj).squeeze(-1)
        
        return x, scores

# ==========================================
# 4. Utilities (Evaluation with NMS)
# ==========================================
def evaluate(model, val_loader, device, iou_threshold=0.5):
    model.eval()
    all_ious = []
    matches = 0
    total_gt = 0
    
    check_limit = 100 # Evaluate more images
    
    for i, (images, gt_boxes, masks) in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
        if i >= check_limit: break
        
        images = images.to(device)
        
        # Sample boxes
        pred_boxes, pred_scores = model.sample(images, num_boxes=50)
        
        # Move to CPU
        pred_boxes = pred_boxes.cpu()
        pred_scores = pred_scores.cpu()
        gt_boxes = gt_boxes.cpu()
        masks = masks.cpu()
        
        for b in range(len(images)):
            # Get valid GT
            valid_gt = gt_boxes[b][masks[b] == 1]
            total_gt += len(valid_gt)
            
            # Get predictions
            boxes = pred_boxes[b]
            scores = pred_scores[b]
            
            # Filter by score
            keep_score = scores > 0.3
            boxes = boxes[keep_score]
            scores = scores[keep_score]
            
            if len(boxes) == 0:
                all_ious.extend([0.0] * len(valid_gt))
                continue
                
            # Apply NMS
            # Convert to [x1, y1, x2, y2] pixel coords for NMS (just scale by 1000 for stability)
            boxes_scaled = boxes * 1000
            keep_nms = nms(boxes_scaled, scores, 0.5)
            boxes = boxes[keep_nms]
            
            if len(valid_gt) == 0: continue
            
            # Hungarian Matching
            iou_matrix = box_iou(valid_gt, boxes) # [M, N]
            cost_matrix = -iou_matrix.numpy()
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                iou = iou_matrix[r, c].item()
                all_ious.append(iou)
                if iou >= iou_threshold:
                    matches += 1
            
            # Unmatched GT
            unmatched_count = len(valid_gt) - len(row_ind)
            all_ious.extend([0.0] * unmatched_count)
            
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    detection_rate = matches / total_gt if total_gt > 0 else 0.0
    
    return mean_iou, detection_rate

# ==========================================
# 5. Main Training Loop
# ==========================================
def main():
    print("Initializing Diffusion Detector Training (Improved)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    base_dir = r"C:\Users\henry-cao-local\OneDrive\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project"
    train_image_dir = os.path.join(base_dir, r"Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Images")
    train_annotation_file = os.path.join(base_dir, r"Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json")
    val_image_dir = os.path.join(base_dir, r"Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Eval_Images")
    val_annotation_file = os.path.join(base_dir, r"Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_val2017.json")
    
    if not os.path.exists(train_image_dir):
        print(f"Error: Train directory not found at {train_image_dir}")
        return

    # Datasets
    print("Loading Datasets...")
    train_dataset = COCOSubsetDataset(train_image_dir, train_annotation_file, transform=transform, subset_size=None)
    val_dataset = COCOSubsetDataset(val_image_dir, val_annotation_file, transform=transform, subset_size=None)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_pad, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_pad, num_workers=0)
    
    # Model
    model = BoxDiffusionModel().to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.cuda.amp.GradScaler()
    
    # Logging
    log_file = "Source/Test_Results/diffusion_training_log.csv"
    os.makedirs("Source/Test_Results", exist_ok=True)
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Mean_IoU', 'Detection_Rate'])
        
    best_det_rate = 0.0
    accumulation_steps = 4
    num_epochs = 50
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for step, (images, boxes, masks) in enumerate(pbar):
            images = images.to(device)
            boxes = boxes.to(device)
            masks = masks.to(device)
            
            with torch.cuda.amp.autocast():
                loss = model(images, boxes, masks)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix(loss=loss.item() * accumulation_steps)
            
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
        
        # Validation
        mean_iou, det_rate = evaluate(model, val_loader, device)
        print(f"Validation Mean IoU: {mean_iou:.4f}, Detection Rate: {det_rate:.4f}")
        
        # Log
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, mean_iou, det_rate])
            
        # Save Best
        if det_rate > best_det_rate:
            best_det_rate = det_rate
            torch.save(model.state_dict(), "Source/Models/best_model_diffusion.pth")
            print("New best model saved!")
            
    print("Training Complete.")

if __name__ == "__main__":
    os.makedirs("Source/Models", exist_ok=True)
    main()
