# Object Detection and OCR Pipeline

## Overview

This project implements a pipeline for detecting and recognizing text within images. The pipeline involves image segmentation, text extraction using OCR, data mapping, and annotation of images with recognized text. The primary goals are:

1. **Object Detection**: Segment and identify objects in the input image.
2. **Text Recognition**: Extract text data from segmented objects using an OCR model.
3. **Data Mapping**: Map extracted data to each object and the master input image.
4. **Annotation**: Output the original image with annotations showing the recognized text and object details.

## Project Structure

## Setup Instructions

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/your-repository.git
    cd your-repository
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. **Install Required Packages**

    ```bash
    pip install -r requirements.txt
    ```

5. **Download Pretrained Models** (if applicable)

    Instructions for downloading or setting up any pretrained models used in the project.

## Usage Guidelines

### 1. Data Preparation

Prepare your data by placing images and labels in the `data/synthetic_text_dataset` directory. Ensure `labels.txt` contains paths and annotations for each image.

Run the data preparation script to generate datasets:

```bash
python scripts/data_preparation.py
python scripts/model_training.py
python scripts/annotation.py --input_image_path path/to/image.png --output_image_path path/to/output_image.png
