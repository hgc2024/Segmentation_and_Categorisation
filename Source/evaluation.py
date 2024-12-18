import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example category mapping for COCO (use a subset if needed)
# COCO 2017 categories
coco_categories = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motorcycle"},
    {"id": 5, "name": "airplane"},
    {"id": 6, "name": "bus"},
    {"id": 7, "name": "train"},
    {"id": 8, "name": "truck"},
    {"id": 9, "name": "boat"},
    {"id": 10, "name": "traffic light"},
    {"id": 11, "name": "fire hydrant"},
    {"id": 13, "name": "stop sign"},
    {"id": 14, "name": "parking meter"},
    {"id": 15, "name": "bench"},
    {"id": 16, "name": "bird"},
    {"id": 17, "name": "cat"},
    {"id": 18, "name": "dog"},
    {"id": 19, "name": "horse"},
    {"id": 20, "name": "sheep"},
    {"id": 21, "name": "cow"},
    {"id": 22, "name": "elephant"},
    {"id": 23, "name": "bear"},
    {"id": 24, "name": "zebra"},
    {"id": 25, "name": "giraffe"},
    {"id": 27, "name": "backpack"},
    {"id": 28, "name": "umbrella"},
    {"id": 31, "name": "handbag"},
    {"id": 32, "name": "tie"},
    {"id": 33, "name": "suitcase"},
    {"id": 34, "name": "frisbee"},
    {"id": 35, "name": "skis"},
    {"id": 36, "name": "snowboard"},
    {"id": 37, "name": "sports ball"},
    {"id": 38, "name": "kite"},
    {"id": 39, "name": "baseball bat"},
    {"id": 40, "name": "baseball glove"},
    {"id": 41, "name": "skateboard"},
    {"id": 42, "name": "surfboard"},
    {"id": 43, "name": "tennis racket"},
    {"id": 44, "name": "bottle"},
    {"id": 46, "name": "wine glass"},
    {"id": 47, "name": "cup"},
    {"id": 48, "name": "fork"},
    {"id": 49, "name": "knife"},
    {"id": 50, "name": "spoon"},
    {"id": 51, "name": "bowl"},
    {"id": 52, "name": "banana"},
    {"id": 53, "name": "apple"},
    {"id": 54, "name": "sandwich"},
    {"id": 55, "name": "orange"},
    {"id": 56, "name": "broccoli"},
    {"id": 57, "name": "carrot"},
    {"id": 58, "name": "hot dog"},
    {"id": 59, "name": "pizza"},
    {"id": 60, "name": "donut"},
    {"id": 61, "name": "cake"},
    {"id": 62, "name": "chair"},
    {"id": 63, "name": "couch"},
    {"id": 64, "name": "potted plant"},
    {"id": 65, "name": "bed"},
    {"id": 67, "name": "dining table"},
    {"id": 70, "name": "toilet"},
    {"id": 72, "name": "tv"},
    {"id": 73, "name": "laptop"},
    {"id": 74, "name": "mouse"},
    {"id": 75, "name": "remote"},
    {"id": 76, "name": "keyboard"},
    {"id": 77, "name": "cell phone"},
    {"id": 78, "name": "microwave"},
    {"id": 79, "name": "oven"},
    {"id": 80, "name": "toaster"},
    {"id": 81, "name": "sink"},
    {"id": 82, "name": "refrigerator"},
    {"id": 84, "name": "book"},
    {"id": 85, "name": "clock"},
    {"id": 86, "name": "vase"},
    {"id": 87, "name": "scissors"},
    {"id": 88, "name": "teddy bear"},
    {"id": 89, "name": "hair drier"},
    {"id": 90, "name": "toothbrush"}
]

# A helper function to get category name from ID
def get_category_name(cat_id):
    for c in coco_categories:
        if c["id"] == cat_id:
            return c["name"]
    return "Unknown"

# Load your model
# Example: Assume model is a Mask R-CNN variant
num_classes = 91  # If using full COCO classes
model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

# Load the saved model weights
model_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Models\best_model_faster_r_cnn_v3.h5"
model.load_state_dict(torch.load(model_path, map_location=device))

# Image transform: same normalization your model expects
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add normalization if your model was trained with it
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def display_predictions(image_path, predictions, score_threshold=0.5, save_path=None):
    """
    Display image with predicted bounding boxes and segmentations.
    predictions is a dict containing 'boxes', 'labels', 'scores', 'masks' (if model is Mask R-CNN).
    """
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    # Convert PIL image to array for mask overlay
    img_np = np.array(img)

    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    masks = predictions.get('masks', None)

    # Iterate over predictions
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        category_name = get_category_name(label.item())
        x1, y1, x2, y2 = box.detach().cpu().numpy()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 10, f"{category_name}: {score:.2f}",
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5)
        )

    # If masks are present (Mask R-CNN), overlay them
    if masks is not None:
        # Masks is a tensor [N, 1, H, W]
        masks_np = (masks > 0.5).squeeze(1).detach().cpu().numpy()
        # Overlay masks
        for mask, label, score in zip(masks_np, labels, scores):
            if score < score_threshold:
                continue
            # Create a colored overlay for mask
            color = (np.random.rand(), np.random.rand(), np.random.rand(), 0.5)
            ax.imshow(np.dstack((mask, mask, mask)) * np.array(color[:3]), alpha=0.5)

    plt.axis('off')
    if save_path:
        # collect the image id from the image path
        image_id = os.path.basename(image_path).split('.')[0]
        save_path = os.path.join(save_path, f"{image_id}_predictions.jpg")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    """
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    """

# Directory containing test images
test_save_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Model_Predictions"

# test_image_path1 = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Eval_Images\000000033221.jpg"  # Replace with an image filename present in test_images_dir

# # Load and preprocess the image
# original_img1 = Image.open(test_image_path1).convert("RGB")
# img_tensor1 = transform(original_img1).to(device)

# # Model expects a list of images
# with torch.no_grad():
#     predictions1 = model([img_tensor1])[0]

# # predictions is a dict: {'boxes', 'labels', 'scores', 'masks'(optional)}
# # Display the results
# display_predictions(test_image_path1, predictions1, score_threshold=0.5, save_path=test_save_dir)

# Test image that doesn't have car or person

test_image_path2 = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\val2017\val2017\000000013923.jpg"

# Load and preproces the image
original_img2 = Image.open(test_image_path2).convert("RGB")
img_tensor2 = transform(original_img2).to(device)

# Model expects a list of images
with torch.no_grad():
    predictions2 = model([img_tensor2])[0]

# Display the results
display_predictions(test_image_path2, predictions2, score_threshold=0.5, save_path=test_save_dir)
