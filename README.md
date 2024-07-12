# DeepFace Detection

## Overview
This project focuses on detecting faces in images using deep learning techniques. The process involves collecting images, annotating them, augmenting the data, training a deep learning model, and implementing object detection through regression and classification. The VGG16 model is utilized, and the losses used are BinaryCrossEntropy and localization loss.

## Steps

### Step 1: Collecting Images Using Webcam
Images are collected using a webcam and saved to a specified directory. The images are captured using OpenCV.

### Step 2: Annotating Images Using LabelMe
The collected images are annotated with bounding boxes using the LabelMe library. This step involves manually drawing bounding boxes around faces in the images.

### Step 3: Data Augmentation Using Albumentations
Data augmentation is performed using the Albumentations library to enhance the dataset. This includes operations like random cropping, flipping, brightness/contrast adjustments, gamma correction, and RGB shifts.

### Step 4: Training the Deep Learning Model
The deep learning model is trained on the augmented dataset. The object detection involves two main tasks:
1. **Regression**: Predicting the coordinates of the bounding boxes.
2. **Classification**: Classifying the image as containing a face (1) or not (0).

### Step 5: Losses
Two types of losses are used during training:
- **BinaryCrossEntropy**: Used for the classification task.
- **Localization Loss**: Used for the regression task to predict bounding box coordinates.

### Step 6: Using VGG16 Model
The VGG16 model, a pre-trained convolutional neural network, is used as the backbone for feature extraction. The model is fine-tuned for the specific task of face detection.

## Installation
To install the required dependencies, run:
```bash
pip install labelme tensorflow tensorflow-gpu opencv-python matplotlib albumentations flask
```

## Usage
1. **Collect Images**:
   Run the script to start collecting images using the webcam.
   ```python
   python collect_images.py
   ```

2. **Annotate Images**:
   Use the LabelMe tool to annotate the collected images with bounding boxes.
   ```bash
   labelme
   ```

3. **Augment Data**:
   Run the script to perform data augmentation on the annotated images.
   ```python
   python augment_data.py
   ```

4. **Train Model**:
   Train the model using the augmented dataset.
   ```python
   python train_model.py
   ```

5. **Deploy with Flask**:
   Deploy the trained model using Flask for real-time face detection.
   ```python
   python app.py
   ```

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests.
