"""AI-assisted medical imaging: chest X-ray classification, Grad-CAM, 3D segmentation.

For education only. Not a medical device and not for clinical use.
"""
import os
from pathlib import Path

import yaml

# MEDIMG_ROOT lets the end-to-end smoke test run the real pipeline in a throwaway directory.
ROOT = Path(os.environ.get("MEDIMG_ROOT", Path(__file__).resolve().parents[2]))


def load_params(path: Path | str = ROOT / "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
