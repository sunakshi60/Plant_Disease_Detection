# Plant Disease Detection using Deep Learning

This project is a deep learning-based system for detecting plant diseases from leaf images using a Convolutional Neural Network (CNN). It aims to help farmers and researchers identify plant health issues early.

---

## Overview

Plant diseases can significantly impact crop yield and quality. Early detection is crucial for timely intervention. This model analyzes leaf images and classifies them into healthy or diseased categories using a trained CNN.

---

## Dataset & Model

- **Dataset**: The model is trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset), which contains over 50,000 images of healthy and diseased leaves across 38 classes.
- **Note**: Please download the dataset manually from Kaggle and place it in the `dataset/` folder before training or running the model.
- **Model**: A custom CNN built using TensorFlow/Keras. The trained `.h5` model is included in the `models/` folder.
