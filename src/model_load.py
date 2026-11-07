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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = r"F:\03 - Projects\8_Signature_Verification_V2\model\NBE_MSSigNet_final.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)


# Model Archtecture
class NBE_MSSigNet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()

        # Shared feature extractor
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
        )

        # Multilevel feature transformations
        self.shallow_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        self.shallow_3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        # Two branches
        self.global_branch = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
        )

        self.regional_branch = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Global-regional channel attention
        self.attention_reduce = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.attention_expand = nn.Sequential(
            nn.Conv2d(32, 256, kernel_size=1),
            nn.Sigmoid(),
        )

        self.projection = nn.Linear(256, embedding_dim)

    def _project(self, feature):
        feature = self.gap(feature).flatten(1)
        feature = F.normalize(feature, p=2, dim=1)
        return self.projection(feature)

    def forward(self, x):
        # Shared features
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        # Branch features
        global_map = self.global_branch(x4)
        regional_map = self.regional_branch(x4)

        # Multilevel fusion
        low_level = self.shallow_2(x2)
        mid_level = self.shallow_3(x3)

        fused_global = low_level * mid_level * global_map
        fused_regional = low_level * mid_level * regional_map

        # Global-regional channel attention
        global_context = self.attention_reduce(self.gap(fused_global))
        regional_context = self.attention_reduce(self.gap(fused_regional))

        joint_context = global_context * regional_context
        channel_gate = self.attention_expand(joint_context)

        attended_global = channel_gate * fused_global
        attended_regional = channel_gate * fused_regional

        # Global representation
        global_embedding = self._project(attended_global)

        # Six regional representations
        horizontal = F.unfold(
            attended_regional,
            kernel_size=(16, 13),
            stride=6
        )
        horizontal = horizontal.view(
            x.size(0), 256, 16, 13, -1
        ).permute(0, 4, 1, 2, 3)

        vertical = F.unfold(
            attended_regional,
            kernel_size=(8, 25),
            stride=4
        )
        vertical = vertical.view(
            x.size(0), 256, 8, 25, -1
        ).permute(0, 4, 1, 2, 3)

        # The chosen CEDAR input geometry produces three regions
        # in each orientation.
        if horizontal.size(1) != 3 or vertical.size(1) != 3:
            raise RuntimeError(
                "Unexpected regional geometry. Check IMAGE_SIZE and "
                "the feature-map dimensions."
            )

        regional_embeddings = []
        for region in horizontal.unbind(dim=1):
            regional_embeddings.append(self._project(region))

        for region in vertical.unbind(dim=1):
            regional_embeddings.append(self._project(region))

        final_embedding = torch.cat(
            [global_embedding] + regional_embeddings,
            dim=1
        )

        return (
            final_embedding,
            global_embedding,
            *regional_embeddings
        )



def get_model():
    model = NBE_MSSigNet(embedding_dim=1024)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model