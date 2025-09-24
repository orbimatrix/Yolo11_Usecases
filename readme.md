# YOLOv11 for Computer Vision Tasks

This README provides a guide on how to use YOLOv11 for various computer vision tasks, including object detection, segmentation, pose estimation, and image classification. The examples are based on the `ultralytics` library.

## Setup

First, you need to install the `ultralytics` library to use YOLOv11 models. This can be done using `pip`:

```bash
!pip install ultralytics
```

## 1\. Object Detection

Object detection involves identifying and localizing objects in an image or video. The `yolo11n.pt` model is pre-trained for this task. You can load the model and perform inference on an image or video.

### Code Example:

```python
from ultralytics import YOLO

# Load a pre-trained object detection model
model = YOLO("yolo11n.pt")

# Run inference on an image
results = model("test.jpg")
results[0].show()

# Run inference on a video and save the results
results = model("cars.mp4", save=True)
results[0].show()
```

## 2\. Segmentation

Segmentation is the process of creating a mask for each object by assigning a label to every pixel in an image. The `yolo11n-seg.pt` model is specifically designed for this task.

### Code Example:

```python
from ultralytics import YOLO

# Load a pre-trained segmentation model
model = YOLO("yolo11n-seg.pt")

# Run inference on an image
results = model("test.jpg")
results[0].show()
```

## 3\. Pose Estimation

Pose estimation identifies the location of key points on an object, such as a person's joints. The `yolo11n-pose.pt` model can be used for this purpose.

### Code Example:

```python
from ultralytics import YOLO

# Load a pre-trained pose estimation model
model = YOLO("yolo11n-pose.pt")

# Run inference on an image
results = model("test.jpg")
results[0].show()
```

## 4\. Image Classification

Image classification assigns a single class label to an entire image. The `yolo11n-cls.pt` model is used for this task.

### Code Example:

```python
from ultralytics import YOLO

# Load a pre-trained classification model
model = YOLO("yolo11n-cls.pt")

# Run inference on an image
results = model("test.jpg")
results[0].show()
```

[Link to Colab](https://colab.research.google.com/drive/1fhuWafPWxZRYVIc6quFb4DmIjpwtHNrM?usp=sharing) 
