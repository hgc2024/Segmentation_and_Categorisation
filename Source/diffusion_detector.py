import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import roi_align
import math
from tqdm import tqdm
import numpy as np

# ==========================================
# 1. Diffusion Noise Scheduler
# ==========================================
class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_boxes, t):
        """
        Forward diffusion process: q(x_t | x_0)
        Adds noise to boxes at timestep t.
        """
        # Ensure t is the right shape for broadcasting
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        noise = torch.randn_like(original_boxes)
        noisy_boxes = sqrt_alpha_bar * original_boxes + sqrt_one_minus_alpha_bar * noise
        return noisy_boxes, noise

# ==========================================
# 2. Time Embedding (Sinusoidal)
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

# ==========================================
# 3. Diffusion Head (The "Denoising" Network)
# ==========================================
class DiffusionHead(nn.Module):
    def __init__(self, feature_dim=256, box_dim=4, time_dim=256):
        super().__init__()
        
        # Time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Main MLP for processing box + features + time
        # Input: [Box Coords (4)] + [Image Features (feature_dim)] + [Time Embed (time_dim)]
        input_dim = box_dim + feature_dim + time_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, box_dim) # Predicts the NOISE, not the box directly
        )

    def forward(self, noisy_boxes, image_features, t):
        # Embed time
        t_emb = self.time_mlp(t) # [Batch, time_dim]
        
        # Concatenate everything
        # We assume image_features are already pooled to [Batch, feature_dim]
        # For a real detector, you'd use RoI Align here on the noisy boxes.
        # To keep it simple and VRAM-light, we'll assume global image context for now
        # or a simplified interaction.
        
        x = torch.cat([noisy_boxes, image_features, t_emb], dim=1)
        return self.layers(x)

# ==========================================
# 4. Full Diffusion Detector Model
# ==========================================
class BoxDiffusionModel(nn.Module):
    def __init__(self, num_classes=2, backbone_name="resnet50"):
        super().__init__()
        
        # A. Backbone (ResNet50)
        # We use a pretrained ResNet to "see" the image.
        print(f"Loading {backbone_name} backbone...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the classification head (fc) and avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze backbone to save VRAM (Critical for 6GB GPU)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Adapter to reduce channels to 256
        self.neck = nn.Conv2d(2048, 256, kernel_size=1)
        
        # B. Diffusion Head
        self.head = DiffusionHead(feature_dim=256, box_dim=4)
        
        # C. Scheduler
        self.scheduler = NoiseScheduler()

    def extract_features(self, images):
        """Extracts features from images using the frozen backbone."""
        with torch.no_grad():
            features = self.backbone(images) # [B, 2048, H/32, W/32]
        
        # Reduce channels
        features = self.neck(features) # [B, 256, H/32, W/32]
        
        # Global Average Pooling to get a single vector per image for conditioning
        # In a full DiffusionDet, you'd use attention over the feature map.
        # Here we use GAP for extreme efficiency.
        global_features = torch.mean(features, dim=[2, 3]) # [B, 256]
        return global_features

    def forward(self, images, gt_boxes=None):
        """
        Training Step:
        1. Extract image features.
        2. Add noise to GT boxes.
        3. Predict the noise.
        4. Calculate loss.
        """
        device = images.device
        batch_size = images.shape[0]
        
        # 1. Image Features
        img_features = self.extract_features(images)
        
        # 2. Sample random timesteps
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=device).long()
        
        # 3. Add noise to GT boxes
        # Note: In a real scenario, you'd pad GT boxes to a fixed number (e.g., 50) per image.
        # Here we assume 1 box per image for the demo/simplicity, or you'd flatten batch * num_boxes.
        # Let's assume input gt_boxes is [Batch, 4] for this simplified implementation.
        noisy_boxes, noise = self.scheduler.add_noise(gt_boxes, t)
        
        # 4. Predict noise
        noise_pred = self.head(noisy_boxes, img_features, t)
        
        # 5. Loss (MSE between predicted noise and actual noise)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, images, num_boxes=5):
        """
        Inference Step (Reverse Diffusion):
        1. Start with random noise boxes.
        2. Iteratively denoise them using the image features.
        """
        device = images.device
        batch_size = images.shape[0]
        
        # 1. Extract features
        img_features = self.extract_features(images)
        
        # 2. Start with pure Gaussian noise
        # Shape: [Batch, 4] (Simplified to 1 box per image for demo)
        # To detect multiple objects, you'd expand this to [Batch * num_boxes, 4]
        # and repeat img_features accordingly.
        x = torch.randn((batch_size, 4), device=device)
        
        # 3. Denoise loop
        for i in tqdm(reversed(range(0, self.scheduler.num_timesteps)), desc="Sampling", leave=False):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.head(x, img_features, t)
            
            # Update x using the reverse diffusion formula (DDPM)
            alpha = self.scheduler.alphas[i]
            alpha_cumprod = self.scheduler.alphas_cumprod[i]
            beta = self.scheduler.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
            
        # Clip boxes to image coordinates (0-1 normalized)
        x = torch.clamp(x, 0, 1)
        return x

# ==========================================
# 5. Utilities: IoU & Evaluation
# ==========================================
def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    Boxes are in [x, y, w, h] format (normalized 0-1).
    """
    # Convert to [x1, y1, x2, y2]
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    x1 = max(b1_x1, b2_x1)
    y1 = max(b1_y1, b2_y1)
    x2 = min(b1_x2, b2_x2)
    y2 = min(b1_y2, b2_y2)

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def evaluate(model, images, gt_boxes):
    """
    Evaluates the model on a batch of images.
    Returns Mean IoU.
    """
    model.eval()
    pred_boxes = model.sample(images)
    
    total_iou = 0
    # Simple evaluation: assume 1-to-1 matching for this demo
    # In a real scenario, you'd use Hungarian matching like in DETR
    for i in range(len(images)):
        iou = compute_iou(pred_boxes[i].cpu().numpy(), gt_boxes[i].cpu().numpy())
        total_iou += iou
        
    return total_iou / len(images)

def save_checkpoint(model, optimizer, epoch, loss, filename="best_model_diffusion.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

# ==========================================
# 6. Demo / Training Loop
# ==========================================
def run_demo():
    print("Initializing Diffusion Detector for 6GB VRAM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    model = BoxDiffusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy Data (Batch of 4 images, 3x512x512)
    # GT Boxes (normalized 0-1): [x_center, y_center, w, h]
    dummy_images = torch.randn(4, 3, 512, 512).to(device)
    dummy_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.3, 0.3, 0.1, 0.1],
        [0.7, 0.7, 0.3, 0.3],
        [0.5, 0.2, 0.1, 0.4]
    ]).to(device)
    
    print("\n--- Starting Training Loop (Gradient Accumulation Demo) ---")
    
    # Gradient Accumulation Settings
    accumulation_steps = 4 
    best_iou = 0.0
    
    # Use Mixed Precision for Memory Savings
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(1, 6): # Run 5 Epochs
        model.train()
        epoch_loss = 0.0
        
        optimizer.zero_grad()
        
        # Simulate a few batches per epoch
        for step in range(8): 
            # Mixed Precision Context
            with torch.cuda.amp.autocast():
                loss = model(dummy_images, dummy_boxes)
                loss = loss / accumulation_steps # Normalize loss
            
            # Backward pass (scaled)
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            
        print(f"Epoch {epoch}: Avg Loss: {epoch_loss/8:.4f}")
        
        # Validation Step
        print(f"Validating Epoch {epoch}...")
        current_iou = evaluate(model, dummy_images, dummy_boxes)
        print(f"Validation Mean IoU: {current_iou:.4f}")
        
        # Save Best Model
        if current_iou > best_iou:
            best_iou = current_iou
            save_checkpoint(model, optimizer, epoch, epoch_loss, filename="Source/Models/best_model_diffusion.pth")
            
    print("\n--- Final Inference (Sampling) ---")
    model.eval()
    predicted_boxes = model.sample(dummy_images)
    print("Predicted Boxes (First 2):")
    print(predicted_boxes[:2])
    print("\nSuccess! The model trains, evaluates, and saves checkpoints.")

if __name__ == "__main__":
    # Ensure Models directory exists
    import os
    os.makedirs("Source/Models", exist_ok=True)
    run_demo()
