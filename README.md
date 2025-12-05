# Traffic Scene Object Detection & Segmentation

## Project Overview
This project focuses on the robust detection and segmentation of **pedestrians and vehicles** in complex traffic environments. Leveraging state-of-the-art deep learning architectures, the system is designed to accurately localize and classify objects even in challenging conditions, such as occlusions or varying lighting.

The core objective was to fine-tune a pre-trained model to achieve high precision in identifying "Person" and "Car" classes, which are critical for autonomous driving and traffic monitoring systems.

## Methodology & Architecture

### Primary Model: Faster R-CNN with ResNet50-FPN
The backbone of this project is the **Faster R-CNN (Region-based Convolutional Neural Network)** architecture, integrated with a **Feature Pyramid Network (FPN)** and a **ResNet-50** backbone.

*   **Backbone (ResNet-50)**: A 50-layer residual network pre-trained on ImageNet is used for feature extraction. This allows the model to leverage learned low-level features (edges, textures) and high-level semantic features effectively.
*   **Feature Pyramid Network (FPN)**: FPN builds a high-level semantic feature map at all scales. This is crucial for detecting objects of varying sizes—small pedestrians vs. large vehicles—by combining low-resolution, semantically strong features with high-resolution, semantically weak features.
*   **Region Proposal Network (RPN)**: Generates candidate object bounding boxes (proposals) which are then refined by the Fast R-CNN detector.

### Training Strategy
*   **Transfer Learning**: We utilized weights pre-trained on the COCO dataset to accelerate convergence and improve generalization on our specific traffic dataset.
*   **Optimization**: The model was trained using the Adam optimizer with a learning rate of `1e-5`, ensuring stable convergence.
*   **Loss Function**: A multi-task loss combining classification loss (Cross Entropy) and regression loss (Smooth L1) for bounding box coordinates.

## Performance Results

The model was evaluated on a held-out test set, achieving the following metrics:

| Metric | Value | Description |
*   OpenCV, NumPy, Pillow

### Running the Model
To train or evaluate the model, navigate to the `Source` directory and run:

```bash
python Source/faster_r_cnn.py
```

Ensure your dataset paths are correctly configured in the `main()` function of `faster_r_cnn.py`.
