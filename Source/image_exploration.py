import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from tqdm import tqdm

# Path to COCO annotation and images
annotation_file = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_train2017.json'
images_dir = r'C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\train2017\train2017'
segmented_dir = r'c:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\panoptic_annotations_trainval2017\annotations\panoptic_train2017\panoptic_train2017'


# Load COCO annotations
with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

    coco_data['images'] = [img for img in coco_data['images']]

# Helper functions
def get_image_info(image_id):
    # Get image info by ID
    return next((img for img in coco_data['images'] if img['id'] == image_id), None)

def get_annotations_for_image(image_id):
    # Get annotations for a specific image
    return [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

def display_image_with_annotations(image_path, annotations, output_path=None):
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
    if output_path:
        annotated_image_name = os.path.basename(image_path)
        fig.savefig(os.path.join(output_path, annotated_image_name))

def display_image_with_segmentations(image_path, annotations, output_path=None):
    # Display an image with its annotations (segmentations)
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    # Add annotations (segmentations)
    for ann in annotations:
        if 'segmentation' in ann and isinstance(ann['segmentation'], list):
            segmentation = ann['segmentation']
            category_id = ann['category_id']
            category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), "Unknown")
            
            for seg in segmentation:
                if len(seg) >= 4:  # Ensure there are at least 2 points (4 coordinates)
                    x_coords = seg[::2]
                    y_coords = seg[1::2]
                    ax.plot(x_coords, y_coords, '-r', linewidth=2)
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        ax.text(
                            x_coords[0], y_coords[0] - 10, category_name,
                            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5)
                        )

    plt.axis('off')
    plt.show()

    # Save the image with annotations
    if output_path:
        annotated_image_name = os.path.basename(image_path)
        fig.savefig(os.path.join(output_path, annotated_image_name))

# # Find 5 images in which at least one of the annotations is a crowd
crowd_images = []
crowd_images_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Crowd_Images"
crowd_image_segmentations_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Crowd_Images\Segmentations"
person_eval_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Eval_Images"
images_eval_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\val2017\val2017"
annotation_eval_file = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Annotations\annotations_trainval2017\annotations\instances_val2017.json"

with open(annotation_eval_file, 'r') as f:
    coco_data_eval = json.load(f)

    coco_data_eval['images'] = [img for img in coco_data_eval['images']]

# store length of images in the dataset
total_images = len(coco_data_eval['images'])

# Please write code so that the actual boxes and labels are displayed for the following image: "C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\val2017\val2017\000000033221.jpg"
# Get the image info and annotations for the specific image
# image_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\val2017\val2017\000000033221.jpg"
image_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Datasets\COCO\Images\val2017\val2017\000000013923.jpg"
image_id = int(os.path.basename(image_path).split('.')[0])
annotations = [ann for ann in coco_data_eval['annotations'] if ann['image_id'] == image_id]
save_path = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Model_Predictions"

# Display the image with annotations
display_image_with_annotations(image_path, annotations, save_path)

# exit()

# Initialize progress bar
with tqdm(total=total_images, desc="Processing images") as pbar:
    for img_info in coco_data_eval['images']:
        annotations = [ann for ann in coco_data_eval['annotations'] if ann['image_id'] == img_info['id']]
        # Check if category_id == 1 for any annotation
        category_ids = [ann['category_id'] for ann in annotations]
        # print(category_ids)
        if 1 in category_ids and 3 in category_ids:
            # Save the image without displaying, don't call display_image_with_annotations
            image_path = os.path.join(images_eval_dir, img_info['file_name'])
            if os.path.isfile(image_path):
                img = Image.open(image_path)
                img.save(os.path.join(person_eval_dir, img_info['file_name']))
            else:
                print(f"Image file not found: {image_path}")

        # Update progress bar
        pbar.update(1)

# exit()

# Please create a subfolder that only contains photos with category ID 1, which is a person
person_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Staging_Area\Segmentation_and_Categorisation\Source\Person_Car_Images"
for img_info in coco_data['images']:
    annotations = get_annotations_for_image(img_info['id'])
    # Check if category_id == 1 for any annotation
    category_ids = [ann['category_id'] for ann in annotations]
    if 1 in category_ids and 3 in category_ids:
        # Save the image without displaying, don't call display_image_with_annotations
        image_path = os.path.join(images_dir, img_info['file_name'])
        if os.path.isfile(image_path):
            img = Image.open(image_path)
            img.save(os.path.join(person_dir, img_info['file_name']))
        else:
            print(f"Image file not found: {image_path}")
        
# exit()

for img_info in coco_data['images']:
    if len(crowd_images) >= 5:
        break
    
    # Get annotations for this image
    annotations = get_annotations_for_image(img_info['id'])
    # Check if iscrowd == 1 for any annotation
    if any(ann['iscrowd'] == 1 for ann in annotations):
        crowd_images.append(img_info)
        # Print category labels for this image
        for ann in annotations:
            category_id = ann['category_id']
            category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), "Unknown")
            print(f"Category ID: {category_id} - Name: {category_name}")
            
        # Save the image with annotations
        img_path = os.path.join(images_dir, img_info['file_name'])
        display_image_with_annotations(img_path, annotations, crowd_images_dir)

        # Save the image with segmentations
        display_image_with_segmentations(img_path, annotations, crowd_image_segmentations_dir)

# Inspect images and annotations
for i, img_info in enumerate(coco_data['images'][:10]):
    print(f"\n--- Image {i + 1} ---")
    print(f"Image ID: {img_info['id']}")
    print(f"File Name: {img_info['file_name']}")
    print(f"Width: {img_info['width']}, Height: {img_info['height']}")

    # Get annotations for this image
    annotations = get_annotations_for_image(img_info['id'])
    print(f"Number of Annotations: {len(annotations)}")

    for ann in annotations[:5]:
        # print id
        print(f"  - Annotation ID: {ann['id']}")
        print(f"  - Category ID: {ann['category_id']}")
        print(f"  - Bounding Box: {ann['bbox']}")
        if 'segmentation' in ann:
            print(f"  - Segmentation: {ann['segmentation'][:1]}...")
        # print area, iscrowd, 
        print(f"  - Area: {ann['area']}")
        print(f"  - Is Crowd: {ann['iscrowd']}")
        
    
    # Display the image with bounding boxes
    image_path = os.path.join(images_dir, img_info['file_name'])
    if os.path.isfile(image_path):
        display_image_with_annotations(image_path, annotations)
    else:
        print(f"Image file not found: {image_path}")