# Facial Impact Project

Facial Impact Project is a real-time facial image processing and beauty enhancement application developed using Python, OpenCV, MediaPipe, and FastAPI.

The project provides a variety of facial editing effects including facial landmark visualization, virtual makeup, eye color simulation, hair color transformation, facial reshaping, aging/de-aging effects, glasses overlays, sticker filters, and real-time camera processing.

## Features

* Facial landmark detection and visualization
* Real-time beauty filters
* Virtual lipstick application
* Eye color (contact lens) simulation
* Hair color transformation
* Face slimming, smile enhancement, and eyebrow adjustment
* Nose enhancement and lip widening
* Aging and de-aging effects
* Glasses and sticker overlays
* Real-time webcam processing

## Technologies

* Python
* FastAPI
* OpenCV
* MediaPipe Face Mesh
* MediaPipe Selfie Multiclass Segmentation
* NumPy
* HTML, CSS, JavaScript

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python -m uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Project Goal

The goal of this project is to explore computer vision, facial landmark analysis, image warping, segmentation, and real-time image processing techniques through an interactive beauty filter studio.
