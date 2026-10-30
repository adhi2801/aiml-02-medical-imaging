from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor(img_u8: np.ndarray, size: int) -> torch.Tensor:
    """uint8 HxW grayscale -> 3xSxS float tensor in [0, 1]."""
    img = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).float().div_(255.0)[None].repeat(3, 1, 1)


def normalize(t: torch.Tensor) -> torch.Tensor:
    return v2.functional.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


def preprocess(img_u8: np.ndarray, size: int) -> torch.Tensor:
    """Exact serve-time transform; training uses the same steps with augmentation in between."""
    return normalize(to_tensor(img_u8, size))


# No horizontal flips: the heart sits on the patient's left, so a flip makes an anatomically
# impossible image. Small rotations/shifts/zooms and exposure changes mimic real acquisition variation.
TRAIN_AUG = v2.Compose([
    v2.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])


class RSNADataset(Dataset):
    def __init__(self, meta: pd.DataFrame, image_dir: Path, img_size: int, train: bool):
        self.meta = meta.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.train = train

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i: int):
        row = self.meta.iloc[i]
        img = cv2.imread(str(self.image_dir / f"{row.patient_id}.png"), cv2.IMREAD_GRAYSCALE)
        x = to_tensor(img, self.img_size)
        if self.train:
            x = TRAIN_AUG(x)  # must run before normalize: ColorJitter clamps to [0, 1]
        return normalize(x), torch.tensor(float(row.label)), i
