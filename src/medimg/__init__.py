"""AI-assisted medical imaging: chest X-ray classification, Grad-CAM, 3D segmentation.

For education only. Not a medical device and not for clinical use.
"""
import os
from pathlib import Path

import yaml

# MEDIMG_ROOT lets the end-to-end smoke test run the real pipeline in a throwaway directory.
ROOT = Path(os.environ.get("MEDIMG_ROOT", Path(__file__).resolve().parents[2]))


def _merge(base: dict, over: dict) -> dict:
    return {**base, **{k: _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
                       for k, v in over.items()}}


def load_params(path: Path | str = ROOT / "params.yaml") -> dict:
    """params.yaml, optionally patched by the YAML file named in MEDIMG_PARAMS (e.g. configs/laptop_gtx1650.yaml).

    The overlay is for running a stage directly on small hardware; `dvc repro` should use plain params.yaml.
    """
    with open(path) as f:
        params = yaml.safe_load(f)
    if overlay := os.environ.get("MEDIMG_PARAMS"):
        with open(overlay) as f:
            params = _merge(params, yaml.safe_load(f) or {})
    return params


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
