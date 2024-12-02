import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Path to COCO annotation and images
annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
images_dir = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\train2017\train2017'

# Load COCO annotations
with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

    # cut it down so that the file name includes "000000391837"
    coco_data['images'] = [img for img in coco_data['images'] if '000000391837' in img['file_name']]

# Helper functions
def get_image_info(image_id):
    # Get image info by ID
    return next((img for img in coco_data['images'] if img['id'] == image_id), None)

def get_annotations_for_image(image_id):
    # Get annotations for a specific image
    return [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

def display_image_with_annotations(image_path, annotations):
    # Display an image with its annotations (bounding boxes)
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    # Add annotations (bounding boxes)
    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), "Unknown")
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            bbox[0], bbox[1] - 10, category_name,
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5)
        )
    
    plt.axis('off')
    plt.show()

    # Save the image with annotations
    img_with_annotation_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Source\Segmentation_and_Categorisation\Source\Sample"
    annotated_image_name = "000000391837_annotated.jpg"
    fig.savefig(os.path.join(img_with_annotation_path, annotated_image_name))

# Inspect images and annotations
for i, img_info in enumerate(coco_data['images'][:3]):
    print(f"\n--- Image {i + 1} ---")
    print(f"Image ID: {img_info['id']}")
    print(f"File Name: {img_info['file_name']}")
    print(f"Width: {img_info['width']}, Height: {img_info['height']}")

    # Get annotations for this image
    annotations = get_annotations_for_image(img_info['id'])
    print(f"Number of Annotations: {len(annotations)}")

    for ann in annotations[:5]:
        print(f"  - Category ID: {ann['category_id']}")
        print(f"  - Bounding Box: {ann['bbox']}")
        if 'segmentation' in ann:
            print(f"  - Segmentation: {ann['segmentation'][:1]}...")
        
    
    # Display the image with bounding boxes
    image_path = os.path.join(images_dir, img_info['file_name'])
    if os.path.isfile(image_path):
        display_image_with_annotations(image_path, annotations)
    else:
        print(f"Image file not found: {image_path}")