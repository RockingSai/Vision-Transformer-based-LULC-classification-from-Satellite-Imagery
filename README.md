# Vision Transformer-Based LULC Classification from Satellite Imagery

## Project Overview

This project uses a **MaxViT (Maximal Vision Transformer)** backbone for image-based **Land Use and Land Cover (LULC)** analysis from satellite imagery. The notebook processes yearly satellite images and generates LULC predictions for the period **2019–2026**.

The project focuses on identifying land-cover classes and analyzing their percentage changes over time.

## Project Details

- **Project Type:** Academic / Final Year Project
- **Domain:** Deep Learning, Computer Vision, Remote Sensing
- **Model:** MaxViT Tiny (`maxvit_tiny_rw_224`)
- **Framework:** PyTorch
- **Language:** Python
- **Platform:** Google Colab
- **Analysis Period:** 2019–2026
- **Team Size:** 4
- **Role:** Team Leader

## LULC Classes

The notebook uses the following eight classes:

1. Water
2. Trees
3. Flooded Vegetation
4. Crops
5. Shrub
6. Built-up
7. Bare Land
8. Snow/Ice

## Tech Stack

- Python
- PyTorch
- PyTorch Image Models (timm)
- NumPy
- Pandas
- OpenCV / PIL
- Matplotlib
- Scikit-learn

## Tools and Platforms

- Google Colab
- Google Drive
- Jupyter Notebook
- PyTorch
- timm

## Methodology

The notebook follows these main steps:

1. Install and import the required Python libraries.
2. Mount Google Drive in Google Colab.
3. Organize satellite images into year-wise folders from 2019 to 2026.
4. Resize input images to **224 × 224** pixels.
5. Apply ImageNet normalization.
6. Load a pretrained **MaxViT Tiny** backbone using `timm`.
7. Add a segmentation head for the eight LULC classes.
8. Generate pixel-level LULC predictions.
9. Calculate the percentage of each land-cover class in the predicted mask.
10. Aggregate the results year-wise.
11. Visualize temporal changes in land cover.

## Model Architecture

The implementation uses:

- **Backbone:** `maxvit_tiny_rw_224`
- **Pretrained weights:** ImageNet pretrained MaxViT backbone
- **Segmentation Head:**
  - 3 × 3 convolution
  - ReLU activation
  - 1 × 1 convolution for eight output classes
- **Output:** 224 × 224 LULC prediction map

## Analysis and Visualizations

The notebook generates:

- Overall LULC change from 2019–2026
- Built-up area / urban expansion analysis
- Water-body change analysis
- Overall vegetation change analysis
- Land-cover composition using stacked area plots
- Net LULC change between 2019 and 2026
- Prediction confidence maps

## Dataset Organization

The notebook expects yearly image folders in Google Drive:

    MyDrive/
    ├── 2019 entire year/
    ├── 2020 entire year/
    ├── 2021 entire year/
    ├── 2022 entire year/
    ├── 2023 entire year/
    ├── 2024 entire year/
    ├── 2025 entire year/
    └── 2026 entire year/

Supported image formats:

- `.png`
- `.jpg`
- `.tif`

## Installation

The notebook is designed for Google Colab.

Install the main model library:

    pip install timm

Required packages:

    pip install torch torchvision numpy pandas matplotlib scikit-learn pillow

## Running the Project

1. Open the `.ipynb` file in Google Colab.
2. Install the required dependencies.
3. Mount Google Drive.
4. Place the yearly satellite images in the expected folders.
5. Run the notebook cells in order.
6. The notebook generates LULC predictions, percentage statistics, visualizations, and confidence maps.

## Project Contributions

As the **Team Leader**, my contributions included:

- Coordinating a team of four members.
- Planning and organizing the project workflow.
- Working on the MaxViT-based LULC inference pipeline.
- Implementing image preprocessing and prediction functions.
- Calculating land-cover percentage statistics.
- Generating temporal LULC analysis and visualizations.
- Contributing to project documentation and presentation.

## Important Note

The current notebook is an **inference and analysis implementation**. It loads an ImageNet-pretrained MaxViT backbone and defines a segmentation head.

The notebook does not contain a training loop for learning the segmentation head from labeled LULC ground-truth data.

The classification-report section uses randomly generated `y_true` and `y_pred` arrays as a placeholder. Therefore, those metrics should **not be interpreted as actual model evaluation results** unless replaced with real validation labels and predictions.

## Project Output

The expected outputs include:

- Year-wise LULC percentage tables
- LULC temporal trend graphs
- Built-up area analysis
- Water-body analysis
- Vegetation change analysis
- Land-cover composition visualization
- Net change analysis
- Prediction confidence maps

## Research Paper

The project is associated with the paper:

**Vision Transformer-Driven Multi-temporal Analysis of Land Use and Land Cover with Satellite Imagery**

Presented at the **International Conference on Networking and Computing Technologies (iCONNECT 2026)**.

## Repository Contents

    .
    ├── Vision_Transformer_based_LULC_classification_from_Satellite_Imagery.ipynb
    └── README.md

## Future Improvements

- Train and fine-tune the segmentation head using labeled LULC data.
- Use a dedicated validation dataset with ground-truth masks.
- Report genuine segmentation metrics such as IoU, Dice score, precision, recall, and F1-score.
- Add automated satellite-image acquisition and preprocessing.
- Improve spatial resolution and boundary accuracy of predicted LULC maps.
- Extend the analysis to additional geographic regions and longer time periods.

---

**Academic Final Year Project | Team Size: 5 | Role: Team Leader**
