# LeukaScan — Lymphocytic Leukemia Detection

A deep-learning web application for detecting lymphocytic leukemia from blood-smear microscopy images.

## Tech Stack
- **Backend:** Python, FastAPI
- **ML Models:** YOLOv8 (cell detection), EfficientNet-B0 CNN (classification)
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Training:** PyTorch, Apple MPS GPU

## Features
- Upload blood smear images
- YOLOv8 cell detection
- CNN classification (89% accuracy)
- Grad-CAM heatmap visualization
- Confidence gauge and charts
- Image quality check
- Analysis history
- PDF report download
- Multi-class detection (ALL, AML, CML)

## Project Structure
leukemia-detection/
├── app/
│   ├── api/          # FastAPI endpoints
│   ├── core/         # Pipeline and dependencies
│   ├── models/       # CNN and YOLO models
│   └── utils/        # Image utils, Grad-CAM, quality check
├── frontend/
│   ├── templates/    # HTML pages
│   └── static/       # CSS and JS
├── scripts/          # Training scripts
├── configs/          # Configuration
└── tests/            # Unit tests

## Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m app.main

## Training
PYTHONPATH=. python scripts/train_cnn.py --epochs 30 --batch 16 --backbone efficientnet_b0 --lr 0.00003

## Disclaimer
This tool is a research prototype and is not a certified medical device.