# Image Super Resolution with ESPCN

This project implements an Efficient Sub-Pixel Convolutional Neural Network (ESPCN)
to upscale low-resolution images using deep learning.

## Features

- Upscaling factors: x2, x3, x4, x8
- TensorFlow implementation
- Web app interface (Flask)
- Comparison with bicubic interpolation
- Trained on DIV2K dataset

## Example

Input image:
60x80

Output (x8):
480x640

## Run locally

Install dependencies:

pip install -r requirements.txt

Run web app:

python app.py

Open:

http://127.0.0.1:5000

## Models

Pretrained models located in:

outputs/checkpoints/

## Dataset

DIV2K dataset:
https://data.vision.ee.ethz.ch/cvl/DIV2K/
