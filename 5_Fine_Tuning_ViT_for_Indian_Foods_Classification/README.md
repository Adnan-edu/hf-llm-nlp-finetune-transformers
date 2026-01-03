# Fine-Tuning ViT for Indian Foods Classification

This project demonstrates fine-tuning a Vision Transformer (ViT) model for the task of classifying various Indian food dishes from images.

## Overview

The project uses the Hugging Face `transformers` and `datasets` libraries to fine-tune a pre-trained Vision Transformer model. The goal is to accurately identify different types of Indian cuisine from a provided image dataset.

## Dataset

- **Name**: `rajistics/indian_food_images` (available on Hugging Face Hub)
- **Content**: Images of various Indian food items such as burger, butter naan, chai, chapati, chole bhature, dal makhani, dhokla, fried rice, idli, jalebi, kaathi rolls, kadai paneer, kulfi, masala dosa, momos, paani puri, pakode, pav bhaji, pizza, and samosa.

## Model

- **Base Model**: `google/vit-base-patch16-224-in21k`
- **Architecture**: Vision Transformer (ViT) pre-trained on ImageNet-21k.
- **Fine-tuning**: The model is fine-tuned with a classification head tailored to the 20 classes of Indian food.

## Workflow

1.  **Environment Setup**: Install necessary libraries (`datasets`, `transformers`, `evaluate`, `accelerate`).
2.  **Data Loading**: Fetch the dataset using `load_dataset`.
3.  **Preprocessing**:
    - Use `AutoImageProcessor` for image normalization and resizing.
    - Apply data augmentation using `torchvision.transforms` (e.g., `RandomResizedCrop`).
4.  **Model Configuration**:
    - Load the pre-trained ViT model.
    - Define label mappings (`id2label`, `label2id`).
5.  **Training**:
    - Set up `TrainingArguments` (learning rate, batch size, epochs).
    - Use the Hugging Face `Trainer` API for the training process.
    - Track metrics like `accuracy`.
6.  **Model Saving**: Save the fine-tuned model for future use.
7.  **Inference**: Use the `pipeline` API to perform predictions on new images.

## Performance Metrics

The model's performance is evaluated using accuracy on the test split. The `Trainer` is configured to load the best model based on accuracy at the end of training.

## Usage

The notebook provides a complete pipeline from loading the data to performing inference on an external image URL. The fine-tuned model can be loaded using the `pipeline("image-classification", model='food_classification')` command.
