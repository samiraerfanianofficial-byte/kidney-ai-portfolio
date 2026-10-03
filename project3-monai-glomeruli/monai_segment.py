# MONAI starter for glomeruli segmentation + explainability
# pip install monai torch torchvision matplotlib

import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize

# 1. Standard MONAI UNet for 2D histology
model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
)

print(model)

# 2. Preprocessing for H&E / PAS slides
transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((256, 256)),
])

print("Preprocessing ready. Add your annotated glomeruli masks for training.")

# 3. Explainability note:
# Use Grad-CAM on decoder features to show why model marks a region as glomerulus.
# This supports WP4 Explainability in KIDNEY-AI proposal.
