"""DVC stage `train_seg`: 3D U-Net (MONAI) for spleen segmentation on CT (Medical Segmentation Decathlon Task09).

Memory budget is designed for small GPUs: 96^3 random patches, AMP, sliding-window validation.
Reports Dice and 95th-percentile Hausdorff distance (HD95, mm) - Dice alone hides boundary errors.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader, PersistentDataset, decollate_batch
from monai.data.utils import json_hashing
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (AsDiscrete, CastToTyped, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped,
                              LoadImaged,
                              Orientationd, RandAffined, RandCropByPosNegLabeld, RandFlipd, RandScaleIntensityd,
                              ScaleIntensityRanged, Spacingd)

from medimg import ROOT, load_params, seed_everything


def transforms(sp: dict, train: bool) -> Compose:
    keys = ["image", "label"]
    base = [
        LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes="RAS"),
        Spacingd(keys, pixdim=sp["pixdim"], mode=("bilinear", "nearest")),
        # Soft-tissue abdominal window in Hounsfield units, mapped to [0, 1].
        ScaleIntensityRanged("image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        # No crop-coordinate keys: they are numpy arrays when fresh and tensors when read from the cache, so a
        # batch mixing both cannot be collated (nothing downstream uses them).
        CropForegroundd(keys, source_key="image", allow_smaller=True, start_coord_key=None, end_coord_key=None),
        # Everything up to here is deterministic and cached on disk; half precision / uint8 keeps the cache small.
        CastToTyped(keys, dtype=(np.float16, np.uint8)),
    ]
    if not train:
        return Compose(base)
    return Compose(base + [
        RandCropByPosNegLabeld(keys, label_key="label", spatial_size=sp["roi_size"], pos=1, neg=1,
                               num_samples=sp["samples_per_volume"], image_key="image", image_threshold=0),
        CastToTyped(keys, dtype=np.float32),
        RandFlipd(keys, prob=0.2, spatial_axis=0),
        RandAffined(keys, prob=0.3, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandScaleIntensityd("image", factors=0.1, prob=0.5),
        # A CT read back from the disk cache arrives as numpy, a freshly processed one as a tensor, and random
        # transforms that do not fire keep the type. Cross-validation mixes both in one batch, so force tensors.
        EnsureTyped(keys, data_type="tensor", dtype=torch.float32),
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

    # DecathlonDataset only downloads and splits; preprocessing is cached on disk by PersistentDataset, because
    # re-resampling every CT each epoch left the GPU idle and an in-RAM cache does not fit a 16 GB laptop.
    common = dict(root_dir=str(root), task=sp["task"], val_frac=0.2, seed=p["seed"], cache_rate=0.0)
    split = {s: DecathlonDataset(section=s, download=(s == "training" and not (root / sp["task"]).exists()),
                                 **common).data for s in ("training", "validation")}
    cache = ROOT / "data" / "processed" / "msd_cache"
    train_ds = PersistentDataset(split["training"], transforms(sp, True), cache_dir=cache / "train",
                                 hash_transform=json_hashing)
    val_ds = PersistentDataset(split["validation"], transforms(sp, False), cache_dir=cache / "val",
                               hash_transform=json_hashing)
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

    best, history, start = {"dice": -1.0}, [], 0
    ckpt = ROOT / "models" / "seg_unet3d.pt"
    ckpt.parent.mkdir(exist_ok=True)
    # Rolling end-of-epoch state so a killed run (OOM, reboot) resumes instead of starting over.
    # Ignored if the params changed, since the LR schedule length depends on max_epochs.
    last = ROOT / "models" / "seg_last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["params"] == sp:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            best, history, start = state["best"], state["history"], state["epoch"] + 1
            print(f"resuming at epoch {start} (best dice so far {best['dice']:.4f})")
        else:
            print("params changed since models/seg_last.pt was written: starting fresh")
    for epoch in range(start, sp["max_epochs"]):
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
                    x, y = batch["image"].to(device).float(), batch["label"].to(device).float()
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
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "history": history, "params": sp}, last.with_suffix(".tmp"))
        last.with_suffix(".tmp").replace(last)
        if run:
            run.log(rec)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "seg_metrics.json").write_text(json.dumps(
        {"val_dice": round(best["dice"], 4), "val_hd95_mm": round(float(np.nan_to_num(best.get("hd95_mm", np.nan))), 2),
         "best_epoch": best.get("epoch"), "n_train": len(train_ds), "n_val": len(val_ds)}, indent=2))
    pd.DataFrame(history).to_csv(out / "seg_history.csv", index=False)
    if run:
        run.summary.update(best)
        run.finish()


if __name__ == "__main__":
    main()
