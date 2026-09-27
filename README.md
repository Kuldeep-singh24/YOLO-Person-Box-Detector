# YOLO Person & Box Object Detection System

## Overview

This project implements a custom YOLO-based object detection and tracking system for detecting two classes:

- Person
- Box

The system supports image detection, video detection, object tracking, object counting, and line-crossing analysis.

A Streamlit web application is also provided for interactive inference.

## Features

- Custom YOLO model training
- Person detection
- Box detection
- Confidence scores
- Image inference
- Video inference
- Object tracking with unique IDs
- Object counting
- Line crossing detection
- Left-to-right counting
- Right-to-left counting
- Streamlit web interface

## Technology Stack

- Python
- YOLO11
- Ultralytics
- OpenCV
- Streamlit
- NumPy
- Pillow

## Dataset

The model was trained using a custom Person and Box dataset containing two classes:

1. box
2. person

Dataset split:

- Training: 1504 images
- Validation: 408 images
- Testing: 217 images

## Model Training

The YOLO11n pretrained model was fine-tuned on the custom dataset.

Training configuration:

- Epochs: 50
- Image size: 640
- Batch size: 8

## Test Results

The model was evaluated on the test dataset.

Overall results:

- Precision: 0.810
- Recall: 0.708
- mAP50: 0.755
- mAP50-95: 0.581

### Box

- Precision: 0.813
- Recall: 0.893
- mAP50: 0.891
- mAP50-95: 0.786

### Person

- Precision: 0.807
- Recall: 0.522
- mAP50: 0.618
- mAP50-95: 0.376

## Project Structure

```text
YOLO_Assignment/
│
├── models/
│   └── best.pt
│
├── dataset/
├── input/
├── output/
├── runs/
│
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── video_detect.py
├── video_tracking.py
├── object_counting.py
├── line_crossing.py
│
├── requirements.txt
└── README.md
###