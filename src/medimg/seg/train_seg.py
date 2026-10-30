"""DVC stage `train_seg`: 3D U-Net (MONAI) for spleen segmentation on CT (Medical Segmentation Decathlon Task09).

Memory budget is designed for small GPUs: 96^3 random patches, AMP, sliding-window validation.
Reports Dice and 95th-percentile Hausdorff distance (HD95, mm) - Dice alone hides boundary errors.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (AsDiscrete, Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandAffined, RandCropByPosNegLabeld, RandFlipd, RandScaleIntensityd, ScaleIntensityRanged,
                              Spacingd)

from medimg import ROOT, load_params, seed_everything


def transforms(sp: dict, train: bool) -> Compose:
    keys = ["image", "label"]
    base = [
        LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes="RAS"),
        Spacingd(keys, pixdim=sp["pixdim"], mode=("bilinear", "nearest")),
        # Soft-tissue abdominal window in Hounsfield units, mapped to [0, 1].
        ScaleIntensityRanged("image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys, source_key="image", allow_smaller=True),
    ]
    if not train:
        return Compose(base)
    return Compose(base + [
        RandCropByPosNegLabeld(keys, label_key="label", spatial_size=sp["roi_size"], pos=1, neg=1,
                               num_samples=sp["samples_per_volume"], image_key="image", image_threshold=0),
        RandFlipd(keys, prob=0.2, spatial_axis=0),
        RandAffined(keys, prob=0.3, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandScaleIntensityd("image", factors=0.1, prob=0.5),
    ])


def build_model() -> UNet:
    return UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH)


def main() -> None:
    p = load_params()
    sp = p["segmentation"]
    seed_everything(p["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = sp["amp"] and device.type == "cuda"
    root = ROOT / sp["root_dir"]
    root.mkdir(parents=True, exist_ok=True)

    common = dict(root_dir=str(root), task=sp["task"], download=not (root / sp["task"]).exists(),
                  val_frac=0.2, seed=p["seed"], cache_rate=sp["cache_rate"], num_workers=sp["num_workers"])
    train_ds = DecathlonDataset(section="training", transform=transforms(sp, True), **common)
    val_ds = DecathlonDataset(section="validation", transform=transforms(sp, False), **{**common, "download": False})
    train_dl = DataLoader(train_ds, batch_size=sp["batch_size"], shuffle=True, num_workers=sp["num_workers"])
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=sp["num_workers"])

    model = build_model().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=sp["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sp["max_epochs"])
    scaler = torch.amp.GradScaler(enabled=amp)
    dice = DiceMetric(include_background=False, reduction="mean")
    hd95 = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    post_pred, post_label = AsDiscrete(argmax=True, to_onehot=2), AsDiscrete(to_onehot=2)

    run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(project="aiml02-seg", config={**sp, "seed": p["seed"]})

    best, history = {"dice": -1.0}, []
    ckpt = ROOT / "models" / "seg_unet3d.pt"
    ckpt.parent.mkdir(exist_ok=True)
    for epoch in range(sp["max_epochs"]):
        model.train()
        epoch_loss, t0 = 0.0, time.time()
        for batch in train_dl:
            x, y = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()
        rec = {"epoch": epoch, "train_loss": epoch_loss / len(train_dl), "epoch_sec": round(time.time() - t0, 1)}

        if (epoch + 1) % sp["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                for batch in val_dl:
                    x, y = batch["image"].to(device), batch["label"].to(device)
                    with torch.autocast(device.type, enabled=amp):
                        out = sliding_window_inference(x, sp["roi_size"], 4, model, overlap=0.5)
                    preds = [post_pred(o) for o in decollate_batch(out)]
                    labels = [post_label(l) for l in decollate_batch(y)]
                    dice(preds, labels)
                    hd95(preds, labels, spacing=sp["pixdim"])
            rec |= {"val_dice": dice.aggregate().item(), "val_hd95_mm": hd95.aggregate().item()}
            dice.reset(), hd95.reset()
            if rec["val_dice"] > best["dice"]:
                best = {"dice": rec["val_dice"], "hd95_mm": rec["val_hd95_mm"], "epoch": epoch}
                torch.save({"state_dict": model.state_dict(), "params": sp}, ckpt)
        history.append(rec)
        print(json.dumps(rec))
        if run:
            run.log(rec)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "seg_metrics.json").write_text(json.dumps(
        {"val_dice": round(best["dice"], 4), "val_hd95_mm": round(float(np.nan_to_num(best.get("hd95_mm", np.nan))), 2),
         "best_epoch": best.get("epoch"), "n_train": len(train_ds), "n_val": len(val_ds)}, indent=2))
    if run:
        run.summary.update(best)
        run.finish()


if __name__ == "__main__":
    main()
