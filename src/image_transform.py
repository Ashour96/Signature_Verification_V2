# ============================================================
# 1. Environment and imports
# ============================================================

import os
import re
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

SEED = 11

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("Device:", device)
print("CUDA available:", torch.cuda.is_available())





IMAGE_HEIGHT = 150
IMAGE_WIDTH = 220

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor()
])



# =========================================
# Orignal Image
# =========================================
def read_image(path, threshold_offset=30, kernel_size=3):
    with Image.open(path) as image:
        # 1. Convert to grayscale
        image = np.array(image.convert("L"))

        # 2. Estimate the most frequent intensity
        histogram = np.bincount(image.ravel(), minlength=256)
        background_value = int(np.argmax(histogram))

        # 3. Convert to strictly black and white
        threshold = max(0, background_value - threshold_offset)

        _, binary = cv2.threshold(
            image,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        # 4. Fill small white gaps in black signature strokes
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        # 5. Guarantee only 0 and 255 remain
        binary = np.where(binary > 0, 255, 0).astype(np.uint8)

        return Image.fromarray(binary, mode="L")


    
def load_signature(image_path):
    """
    Load and preprocess a signature image.

    Returns:
        Tensor with shape (1, 1, 150, 220)
    """
    # image = Image.open(image_path).convert("L")
    image = read_image(image_path)

    # # Make images slightly darker
    # brightness_factor = 0.015  # 1.0 = original, < 1.0 = darker
    # image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    
    # convert to tensor
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor




# =========================================
# Cropped Image
# =========================================

# def read_image_cropped_image(image, threshold_offset=30, kernel_size=3):
#     """
#     Convert a PIL image into a strictly black-and-white signature image.

#     Args:
#         image: PIL Image object.
#         threshold_offset: Offset below the estimated background intensity.
#         kernel_size: Morphological closing kernel size.

#     Returns:
#         PIL grayscale image containing only 0 and 255.
#     """
#     # 1. Convert the cropped image to grayscale
#     image = np.array(image.convert("L"))

#     # 2. Estimate the most frequent intensity (background)
#     histogram = np.bincount(image.ravel(), minlength=256)
#     background_value = int(np.argmax(histogram))

#     # 3. Convert to strictly black and white
#     threshold = max(0, background_value - threshold_offset)

#     _, binary = cv2.threshold(
#         image,
#         threshold,
#         255,
#         cv2.THRESH_BINARY
#     )

#     # 4. Fill small white gaps within signature strokes
#     kernel = cv2.getStructuringElement(
#         cv2.MORPH_ELLIPSE,
#         (kernel_size, kernel_size)
#     )

#     binary = cv2.morphologyEx(
#         binary,
#         cv2.MORPH_CLOSE,
#         kernel,
#         iterations=1
#     )

#     # 5. Guarantee only black (0) and white (255)
#     binary = np.where(binary > 0, 255, 0).astype(np.uint8)

#     return Image.fromarray(binary, mode="L")



# def load_signature_cropped_image(image, threshold_offset):
#     """
#     Preprocess a cropped PIL signature image.

#     Returns:
#         Tensor with shape (1, 1, 150, 220).
#     """
#     # 1. Binarize the cropped image
#     image = read_image_cropped_image(image, threshold_offset)

#     # 2. Resize to the model's expected dimensions
#     image = image.resize((220, 150), Image.Resampling.LANCZOS)

#     # 3. Convert to tensor using your existing transform
#     image_tensor = transform(image)

#     # 4. Add batch dimension
#     image_tensor = image_tensor.unsqueeze(0)

#     return image_tensor

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################


def read_image_cropped_image(
    image,
    brightness=1.0,
    contrast=1.0,
    threshold_offset=30,
    kernel_size=3
):
    """
    Convert a cropped PIL image into a strictly black-and-white
    signature image after brightness and contrast adjustments.

    Args:
        image: PIL Image object.
        brightness: Brightness factor (1.0 = original).
        contrast: Contrast factor (1.0 = original).
        threshold_offset: Offset below estimated background intensity.
        kernel_size: Morphological closing kernel size.

    Returns:
        PIL grayscale image containing only 0 and 255.
    """

    # 1. Convert to grayscale
    image = image.convert("L")

    # 2. Adjust brightness
    image = ImageEnhance.Brightness(image).enhance(brightness)

    # 3. Adjust contrast
    image = ImageEnhance.Contrast(image).enhance(contrast)

    # 4. Convert to NumPy array
    image = np.array(image)

    # 5. Estimate the most frequent intensity (background)
    histogram = np.bincount(image.ravel(), minlength=256)
    background_value = int(np.argmax(histogram))

    # 6. Calculate threshold
    threshold = max(0, background_value - threshold_offset)

    # 7. Convert to strictly black and white
    _, binary = cv2.threshold(
        image,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # 8. Fill small gaps within signature strokes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # 9. Guarantee only black (0) and white (255)
    binary = np.where(binary > 0, 255, 0).astype(np.uint8)

    return Image.fromarray(binary, mode="L")


def load_signature_cropped_image(
    image,
    brightness=1.0,
    contrast=1.0,
    threshold_offset=30
):
    """
    Preprocess a cropped PIL signature image.

    Returns:
        Tensor with shape (1, 1, 150, 220).
    """

    # 1. Preprocess image
    image = read_image_cropped_image(
        image,
        brightness=brightness,
        contrast=contrast,
        threshold_offset=threshold_offset
    )

    # 2. Resize
    image = image.resize(
        (220, 150),
        Image.Resampling.LANCZOS
    )

    # 3. Convert to tensor
    image_tensor = transform(image)

    # 4. Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor