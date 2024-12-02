# Import statements
import os
from PIL import Image

# Pre=processing function
# Using the given image, the function applies various preprocessing steps to the image
# It saves the preprocessed image to the output path
def preprocessing_image(image_path, output_path, s3_bucket=None, s3_key=None):
    try:
        # Load image
        img = Image.open(image_path)

        # Apply pre-processing, to be decided later

        # Save the pre-processed image
        img.save(output_path)
        print(f"Pre-processed image saved to: {output_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

"""
import boto3
from botocore.exceptions import NoCredentialsError

# Pre-processing function
# Using the given image, the function applies various preprocessing steps to the image
# It saves the preprocessed image to the output path or uploads it to S3 if specified
def preprocessing_image(image_path, output_path, s3_bucket=None, s3_key=None):
    try:
        # Load image
        img = Image.open(image_path)

        # Apply pre-processing, to be decided later

        # Save the pre-processed image locally
        img.save(output_path)
        print(f"Pre-processed image saved to: {output_path}")

        # If S3 bucket and key are provided, upload the image to S3
        if s3_bucket and s3_key:
            s3 = boto3.client('s3')
            try:
                s3.upload_file(output_path, s3_bucket, s3_key)
                print(f"Pre-processed image uploaded to S3: s3://{s3_bucket}/{s3_key}")
            except FileNotFoundError:
                print(f"The file {output_path} was not found")
            except NoCredentialsError:
                print("Credentials not available")
"""

# Directory image processing function
# Using the given input directory, the function processes all images in the directory
# The images are stored in the output directory
# This function is integrated with Amazon S3
def process_images_in_directory(input_dir, output_dir, s3_bucket=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('jpg', 'jpeg', 'png')):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            s3_key = f"preprocessed/{file_name}" if s3_bucket else None
            preprocessing_image(input_path, output_path, s3_bucket, s3_key)


if __name__ == "__main__":
    # Define input and output directories
    input_images_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Source\Segmentation_and_Categorisation\Source\Test_Images"
    output_images_dir = r"C:\Users\henry-cao-local\Desktop\Self_Learning\Computer_Vision_Engineering\Segmentation_Project\Source\Segmentation_and_Categorisation\Source\Test_Output"

    # Define S3 buckets (optional)
    s3_bucket = None

    # Process image s in the directory
    process_images_in_directory(input_images_dir, output_images_dir, s3_bucket)