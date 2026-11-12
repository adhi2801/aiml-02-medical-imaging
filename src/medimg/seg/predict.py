"""Segment the spleen in one abdominal CT on a CPU, with the released cross-validation models.

    python -m medimg.seg.predict demo/spleen_ct.nii.gz                 # all five fold models as an ensemble
    python -m medimg.seg.predict ct.nii.gz --models models/seg_cv_fold0.pt --out out/

Writes, next to each other in --out:
  <name>_spleen.nii.gz   the mask, resampled back onto the input CT's own grid (opens over the CT in any viewer)
  <name>_spleen.json     volume in mL, how well the models agree, runtime
  <name>_spleen.png      axial and coronal slices through the largest cross-section

Each fold model was trained on about 33 of the 41 labelled CTs, so for a new CT the five are independent opinions.
Where they disagree is reported (`fold_agreement_dice`): low agreement means "check this one by hand", which is the
signal a reviewer needs and a single model cannot give. The cross-validation showed when that happens: enlarged or
weakly enhanced spleens (README, failure analysis).

For education only. Not a medical device.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import MetaTensor, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, CropForegroundd, EnsureChannelFirstd, Invertd, LoadImaged, Orientationd,
                              SaveImaged, ScaleIntensityRanged, Spacingd)

from medimg import ROOT, load_params
from medimg.seg.figures_seg import keep_largest
from medimg.seg.train_seg import build_model


def preprocessing(pixdim) -> Compose:
    """The evaluation transforms of train_seg, for an image without a label."""
    return Compose([
        LoadImaged("image"), EnsureChannelFirstd("image"), Orientationd("image", axcodes="RAS"),
        Spacingd("image", pixdim=pixdim, mode="bilinear"),
        ScaleIntensityRanged("image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd("image", source_key="image", allow_smaller=True, start_coord_key=None, end_coord_key=None),
    ])


def load_models(paths: list[Path], device) -> list[torch.nn.Module]:
    models = []
    for p in paths:
        net = build_model()
        state = torch.load(p, map_location=device, weights_only=False)
        net.load_state_dict(state["state_dict"])
        models.append(net.to(device).eval())
    return models


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(a, b).sum() / s)


def fold_agreement(masks: list[np.ndarray]) -> float | None:
    """Mean pairwise Dice between the individual models' masks (None with a single model)."""
    pairs = list(itertools.combinations(masks, 2))
    return round(float(np.mean([dice(a, b) for a, b in pairs])), 4) if pairs else None


def segment(image: torch.Tensor, models: list[torch.nn.Module], roi, sw_batch: int = 2):
    """Ensemble softmax over the models, then the largest connected component. Returns (mask, per-model masks)."""
    probs, singles = [], []
    with torch.no_grad():
        for net in models:
            p = torch.softmax(sliding_window_inference(image, roi, sw_batch, net, overlap=0.5), dim=1)[:, 1:2]
            probs.append(p)
            singles.append(keep_largest((p[0, 0] > 0.5).cpu().numpy().astype(np.uint8)))
    mean = torch.stack(probs).mean(0)
    return keep_largest((mean[0, 0] > 0.5).cpu().numpy().astype(np.uint8)), singles


def save_figure(ct: np.ndarray, mask: np.ndarray, path: Path, title: str, pixdim) -> None:
    """Radiological convention: arrays are RAS, so x is flipped to put the patient's right on the viewer's left."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct, mask = ct[::-1], mask[::-1]
    z = int(mask.sum(axis=(0, 1)).argmax()) if mask.any() else ct.shape[2] // 2
    y = int(mask.sum(axis=(0, 2)).argmax()) if mask.any() else ct.shape[1] // 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a, img, m, name, aspect in ((ax[0], ct[:, :, z].T, mask[:, :, z].T, f"axial z={z}", pixdim[1] / pixdim[0]),
                                    (ax[1], ct[:, y, :].T, mask[:, y, :].T, f"coronal y={y}", pixdim[2] / pixdim[0])):
        a.imshow(img, cmap="gray", origin="lower", aspect=aspect)
        a.contour(m, levels=[0.5], colors="lime", linewidths=1.2)
        a.set_title(name)
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ct", type=Path, help="abdominal CT as NIfTI (.nii or .nii.gz), in Hounsfield units")
    ap.add_argument("--models", type=Path, nargs="+",
                    default=sorted((ROOT / "models").glob("seg_cv_fold*.pt")))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if not args.models:
        raise SystemExit("no models: run `python scripts/fetch_models.py` first")

    sp = load_params()["segmentation"]
    device = torch.device(args.device)
    name = args.ct.name.removesuffix(".gz").removesuffix(".nii")
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pre = preprocessing(sp["pixdim"])
    item = pre({"image": str(args.ct)})
    image = item["image"][None].to(device).float()
    mask, singles = segment(image, load_models(args.models, device), sp["roi_size"])
    seconds = time.perf_counter() - t0

    # Back onto the input's own grid, so the mask overlays the original CT in any viewer.
    item["pred"] = MetaTensor(torch.from_numpy(mask)[None].float(), meta=item["image"].meta)
    inverted = Invertd("pred", transform=pre, orig_keys="image", nearest_interp=True)(item)
    SaveImaged("pred", output_dir=args.out, output_postfix="spleen", separate_folder=False, resample=False,
               output_dtype=np.uint8, print_log=False)(inverted)

    voxel_ml = float(np.prod(sp["pixdim"])) / 1000.0
    summary = {
        "ct": args.ct.name, "models": [p.name for p in args.models],
        "spleen_volume_ml": round(float(mask.sum()) * voxel_ml, 1),
        "fold_agreement_dice": fold_agreement(singles),
        "per_model_volume_ml": [round(float(m.sum()) * voxel_ml, 1) for m in singles],
        "seconds": round(seconds, 1), "device": str(device),
        "note": "Education only. Not a medical device.",
    }
    (args.out / f"{name}_spleen.json").write_text(json.dumps(summary, indent=2))
    title = f"{name}: spleen {summary['spleen_volume_ml']} mL"
    if summary["fold_agreement_dice"] is not None:
        title += f", models agree (Dice) {summary['fold_agreement_dice']:.2f}"
    save_figure(image[0, 0].cpu().numpy(), mask, args.out / f"{name}_spleen.png", title, sp["pixdim"])
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
