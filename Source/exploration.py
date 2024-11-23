import time
from pycocotools.coco import COCO
import torch
import tensorflow
from roboflow import Roboflow
from roboflow.adapters.rfapi import RoboflowError
import requests
from requests.exceptions import RequestException

def retry_operation(operation, max_retries=10, retry_delay=5):
    """Generic retry function for handling API operations"""
    for attempt in range(max_retries):
        try:
            result = operation()
            print(f"Operation completed successfully")
            return result
        except (RoboflowError, RequestException, ConnectionError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting.")
                raise

# Load COCO annotations
annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
coco = COCO(annotation_file)

# Get all categories
categories = coco.loadCats(coco.getCatIds())
category_names = [cat['name'] for cat in categories]
print("COCO Categories: ", category_names)

# Get image ID's for a specific category
cat_ids = coco.getCatIds(catNms=['person'])
img_ids = coco.getImgIds(catIds=cat_ids)
img_ids = img_ids[:20]  # For training, just use 20 images

# Initialize Roboflow
rf = retry_operation(lambda: Roboflow(api_key="uabftMXJfyak881AxBvr"))

project_id = "object_identification-psgqy"
# Access the project
project = retry_operation(lambda: rf.workspace().project(project_id))

# Load and pre-process the COCO images
for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_path = fr"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\train2017\train2017\{img_info['file_name']}"
    
    # Upload image with retry mechanism
    retry_operation(lambda: project.upload(img_path))

# Define preprocessing and augmentation settings
settings = {
    "preprocessing": {
        "resize": {
            "width": 640,
            "height": 640,
            "format": "Stretch to"
        },
        "auto-orient": True,
        "contrast": {"type": "Contrast Stretching"},
        "brightness": {"percent": 10}
    },
    "augmentation": {
        "flip": {"horizontal": True, "vertical": False},
        "rotate": {"degrees": 15}
    }
}

# Generate a new version with the specified settings
new_version = retry_operation(lambda: project.generate_version(settings=settings))
print(f"New version created")


