import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Simple U-Net block for kidney glomeruli segmentation
# Dataset: HuBMAP Kidney (open) - download tiles and masks separately
# This is a starter template to show training pipeline

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = DoubleConv(3, 32)
        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(32, 64)
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        e = self.enc(x)
        p = self.pool(e)
        m = self.mid(p)
        u = self.up(m)
        d = self.dec(torch.cat([u, e], dim=1))
        return torch.sigmoid(self.out(d))

def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2*inter + eps) / (pred.sum() + target.sum() + eps)

# Dummy run to prove pipeline works
model = TinyUNet()
x = torch.randn(2, 3, 128, 128)
y = torch.randint(0, 2, (2, 1, 128, 128)).float()
pred = model(x)
print(f"Dice on dummy batch: {dice_score(pred, y):.3f}")
print("Replace dummy data with HuBMAP tiles for real training")
