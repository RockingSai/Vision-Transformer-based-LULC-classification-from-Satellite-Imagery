# Vision-Transformer-based-LULC-classification-from-Satellite-Imagery

# PyTorch Image Models
!pip install timm

import torch
import timm
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import os

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# ============================================================
# 1. GOOGLE DRIVE
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = "/content/drive/MyDrive"


# ============================================================
# 2. DATASET FOLDERS
# ============================================================

YEAR_FOLDERS = {
    "2019": "2019 entire year",
    "2020": "2020 entire year",
    "2021": "2021 entire year",
    "2022": "2022 entire year",
    "2023": "2023 entire year",
    "2024": "2024 entire year",
    "2025": "2025 entire year",
    "2026": "2026 entire year"
}


# ============================================================
# 3. LULC CLASSES
# ============================================================

CLASSES = [
    "Water",
    "Trees",
    "Flooded_Veg",
    "Crops",
    "Shrub",
    "Built_up",
    "Bare_land",
    "Snow_Ice"
]

NUM_CLASSES = len(CLASSES)


# ============================================================
# 4. MAXVIT SEGMENTATION MODEL
# ============================================================

class MaxViTSegmentation(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.backbone = timm.create_model(
            "maxvit_tiny_rw_224",
            pretrained=True,
            features_only=True
        )

        in_channels = self.backbone.feature_info[-1]["num_chs"]

        self.seg_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                256,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, x):

        features = self.backbone(x)

        x = features[-1]

        x = self.seg_head(x)

        x = nn.functional.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        return x


# ============================================================
# 5. LOAD MODEL
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MaxViTSegmentation(NUM_CLASSES)

model = model.to(device)

model.eval()

print("Using device:", device)


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

transform = T.Compose([

    T.Resize((224, 224)),

    T.ToTensor(),

    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. LULC PREDICTION
# ============================================================

def predict_lulc(image_path):

    img = Image.open(image_path).convert("RGB")

    input_tensor = transform(img)

    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

    pred = torch.argmax(
        output,
        dim=1
    ).squeeze().cpu().numpy()

    return pred


# ============================================================
# 8. CALCULATE LULC PERCENTAGES
# ============================================================

def calculate_percentages(mask):

    total_pixels = mask.size

    percentages = {}

    for i, cls in enumerate(CLASSES):

        percentages[cls] = (
            np.sum(mask == i) /
            total_pixels
        ) * 100

    return percentages


# ============================================================
# 9. YEAR-WISE LULC ANALYSIS
# ============================================================

yearly_results = {}

for year, folder in YEAR_FOLDERS.items():

    folder_path = os.path.join(
        BASE_PATH,
        folder
    )

    if not os.path.exists(folder_path):

        print(
            f"Folder not found: {folder_path}"
        )

        continue

    images = [
        os.path.join(folder_path, f)

        for f in os.listdir(folder_path)

        if f.lower().endswith(
            (".png", ".jpg", ".tif")
        )
    ]

    if len(images) == 0:

        print(
            f"No images found for {year}"
        )

        continue

    yearly_sum = Counter()

    for img_path in images:

        mask = predict_lulc(
            img_path
        )

        percentages = calculate_percentages(
            mask
        )

        yearly_sum.update(
            percentages
        )

    average = {
        key: value / len(images)

        for key, value
        in yearly_sum.items()
    }

    yearly_results[year] = average


# ============================================================
# 10. RESULTS TABLE
# ============================================================

df = pd.DataFrame(
    yearly_results
).T.round(2)

print(df)


# ============================================================
# 11. OVERALL LULC CHANGE
# ============================================================

plt.figure(figsize=(12, 6))

for cls in df.columns:

    plt.plot(
        df.index,
        df[cls],
        marker="o",
        label=cls
    )

plt.xlabel("Year")

plt.ylabel(
    "Area Percentage (%)"
)

plt.title(
    "MaxViT-based LULC Change Analysis (2019–2026)"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 12. BUILT-UP AREA / URBAN EXPANSION
# ============================================================

plt.figure(figsize=(7, 5))

plt.bar(
    df.index,
    df["Built_up"]
)

plt.title(
    "Urban Expansion using MaxViT"
)

plt.ylabel(
    "Built-up Area (%)"
)

plt.show()


# ============================================================
# 13. WATER CHANGE - LINE GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df.index,
    df["Water"],
    marker="o"
)

plt.xlabel("Year")

plt.ylabel(
    "Water Area (%)"
)

plt.title(
    "Temporal Change in Water Bodies (2019–2026)"
)

plt.grid(True)

plt.show()


# ============================================================
# 14. WATER CHANGE - BAR GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    df.index,
    df["Water"]
)

plt.xlabel("Year")

plt.ylabel(
    "Water Area (%)"
)

plt.title(
    "Year-wise Water Body Percentage"
)

plt.show()


# ============================================================
# 15. TOTAL VEGETATION
# ============================================================

df["Total_Vegetation"] = (

    df["Trees"]

    + df["Shrub"]

    + df["Crops"]

    + df["Flooded_Veg"]
)


plt.figure(figsize=(8, 5))

plt.plot(
    df.index,
    df["Total_Vegetation"],
    marker="o"
)

plt.xlabel("Year")

plt.ylabel(
    "Vegetation Area (%)"
)

plt.title(
    "Overall Vegetation Change (2019–2026)"
)

plt.grid(True)

plt.show()


# ============================================================
# 16. LAND COVER COMPOSITION
# ============================================================

plt.figure(figsize=(12, 6))

plt.stackplot(

    df.index,

    df["Water"],

    df["Total_Vegetation"],

    df["Built_up"],

    df["Bare_land"],

    labels=[
        "Water",
        "Vegetation",
        "Built-up",
        "Bare land"
    ]
)

plt.legend(
    loc="upper left"
)

plt.xlabel("Year")

plt.ylabel(
    "Area Percentage (%)"
)

plt.title(
    "Land Cover Composition Change Over Time"
)

plt.show()


# ============================================================
# 17. NET LULC CHANGE
# ============================================================

change_df = (
    df.loc["2026"]
    - df.loc["2019"]
)

plt.figure(figsize=(8, 5))

plt.bar(
    change_df.index,
    change_df.values
)

plt.xticks(
    rotation=45
)

plt.ylabel(
    "Percentage Change"
)

plt.title(
    "Net LULC Change (2019–2026)"
)

plt.grid(axis="y")

plt.tight_layout()

plt.show()


# ============================================================
# 18. PREDICTION WITH CONFIDENCE
# ============================================================

def predict_with_confidence(image_path):

    img = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        img
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probs = torch.softmax(
            output,
            dim=1
        )

    confidence = torch.max(
        probs,
        dim=1
    )[0].squeeze().cpu().numpy()

    prediction = torch.argmax(
        probs,
        dim=1
    ).squeeze().cpu().numpy()

    return prediction, confidence


# ============================================================
# 19. DISPLAY CONFIDENCE MAP
# ============================================================

if "images" in locals() and len(images) > 0:

    pred, confidence = (
        predict_with_confidence(
            images[0]
        )
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(
        confidence,
        cmap="viridis"
    )

    plt.colorbar(
        label="Confidence Score"
    )

    plt.title(
        "MaxViT Prediction Confidence Map"
    )

    plt.axis("off")

    plt.show()
