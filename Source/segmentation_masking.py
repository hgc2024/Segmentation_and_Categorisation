# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Define  the backbone network (ResNet-like)
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__() # Call the parent class constructor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3) # A convolutional layer with 3 input channels, 64 output channels, 7x7 kernel, stride 2, and padding 3)
        self.bn1 = nn.BatchNorm2d(64) # A batch normalisation layer
        self.relu = nn.ReLU(inplace=True) # A ReLU activation function
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # A max pooling layer with 3x3 kernel, stride 2, and padding 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # A convolutional layer with 64 input channels, 128 output channels, 3x3 kernel, stride 1, and padding 1
            nn.BatchNorm2d(128), # A batch normalisation layer
            nn.ReLU(inplace=True), # A ReLU activation function
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # A convolutional layer with 128 input channels, 256 output channels, 3x3 kernel, stride 1, and padding 1
            nn.BatchNorm2d(256), # A batch normalisation layer
            nn.ReLU(inplace=True), # A ReLU activation function
        )
    
    # Forward pass function
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Define the RPN (Region Proposal Network)
class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1) # A convolutional layer with in_channels input channels, 512 output channels, 3x3 kernel, stride 1, and padding 1
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1, stride=1) # A convolutional layer with 512 input channels, num_anchors * 2 output channels, 1x1 kernel, stride 1, and pading 1
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg

# Define the Mask R-CNN model
class MaskRCNN(nn.Module):
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.backbone = Backbone()
        self.rpn = RPN(in_channels=256, num_anchors=9) # 256 input channels from the backbone network, 9 anchors per spatial location
        self.roi_align = nn.AdaptiveMaxPool2d((7, 7)) # Adaptive max pooling layer to resize the region of interest (RoI) to a fixed size
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024) # A fully connected layer to reduce the feature dimensionality
            nn.ReLU(), # A ReLU activation function
            nn.Linear(1024, 1024), # A fully connected layer
            nn.ReLU(), # A ReLU activation function
            nn.Linear(1042, 21) # A fully connected layer with 21 output claases (20 classes + background)
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # A convolutional layer with 256 input channels, 256 output channels, 3x3 kernel, and padding 1
            nn.ReLU(), # A ReLU activation function
            nn.Conv2d(256, 1, kernel_size=1), # A convolutional layer with 256 input channels, 1 output channel, and 1x1 kernel (binary mask output)
        )
    
    # Foward pass function
    def forward(self, images, targets=None):
        features = self.backbone(images) # Backbone network
        rpn_logits, rpn_bbox = self.rpn(features) # Region Proposan Network
        proposals = self.generate_proposals(rpn_logits, rpn_bbox) # Generate region proposals
        roi_pooled = self.roi_align(features, proposals) # Resize region of interest
        roi_flattened = roi_pooled.view(roi_pooled.size(0), -1)
        class_logits = self.classifier(roi_flattened)
        mask_logits = self.mask_head(features)
        return class_logits, mask_logits
    
    # Proposal generation function
    def generate_proposals(self, rpn_logits, rpn_bbox):
        pass # Placeholder for proposal generation

# Load and prepare the dataset
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir # Root directory
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')] # Creates list of image files, which must end with '.jpg'
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx]) # Image path
        img = cv2.imread(img_path) # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image to RGB

        # Retrieve annotations based on the index
        annotation_file = "" # Placeholder, to be implemented
        annotations = self.get_annotations(annotation_file, idx)
        target = {
            'category_id': torch.tensor(annotations['category_id']),
            'boxes': torch.tensor(annotations['bbox']),
            'segmentations': torch.tensor(annotations['segmentation'])
        }

        if self.transforms:
            img = self.transforms(img)
        
        return img, target

# Dataset and DataLoader initialisation
root_dir = r""
dataset = CocoDataset(root_dir, transforms=TF.to_tensor) # Create dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=10) # Create DataLoader

# Initialise the Mask R-CNN model
model = MaskRCNN()
model.to(device)

# Define optimiser and learning rate
params = [p for p in model.parameters() if p.requires_grad] # Parameters to be optimised
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005) # Stochastic Gradient Descent (SGD) optimiser

# Training loop function
def train_one_epoch(model, optimizer, data_loader, device):
    model.train() # Set model to training mode
    for images, targets in data_loader:
        images = [img.to(device) for img in images] # Move images to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # Move targets to device

        class_logits, mask_logits = model(images) # Foward pass

        classification_loss = compute_classification_loss(class_logits, targets)
        mask_loss = compute_mask_loss(mask_logits, targets)
        losses = classification_loss + mask_loss

        optimizer.zero_grad() # Zero the gradients
        losses.backward() # Backward pass
        optimizer.step() # Optimiser step
    
    print(f"Loss: {losses.item()}")

# Classification loss function
def compute_classification_loss(class_logits, targets):
    pass # Placeholder for classification loss computation

# Mask loss function
def compute_mask_loss(mask_logits, targets):
    pass # Placeholder for mask loss computation

num_epochs = 10 # Number of epochs
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch(model, optimizer, dataloader, device) # Train for one epoch

# Inference function
def predict(image_path, model, device):
    model.eval() # Set model to evaluation mode
    img = cv2.imread(image_path) # Read image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image to RGB
    img_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(device) # Convert image to tensor and move to device

    with torch.no_grad():
        class_logits, mask_logits = model(img_tensor) # Forward pass
    
    # Visualise the results with actual bounding box and mask visualisation
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_rgb)

    # Add bounding boxes
    for mask in mask_logits:
        mask = mask.cpu().numpy().squeeze() # Convert mask to numpy array
        mask = mask > 0.5
        ax.imshow(mask, alpha=0.05)

        plt.axis('off')
        plt.show()

        # Save the image with bounding boxes
        img_with_mask_path = r""
        annotated_image_name = "annotated_image.jpg"
        fig.savefig(os.path.join(img_with_mask_path, annotated_image_name))

# Test the model with an image
sample_image_path = r""
predict(sample_image_path, model, device)