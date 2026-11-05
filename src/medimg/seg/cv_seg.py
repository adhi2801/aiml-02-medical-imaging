"""5-fold cross-validation of the spleen 3D U-Net over all 41 labelled MSD Task09 CTs, with no checkpoint selection.

    python -m medimg.seg.cv_seg            (Kaggle: `python scripts/kaggle_push.py --seg-cv`)

Why: with a single 33/8 split, the "best epoch" is picked on the same 8 CTs it is scored on. That gave Dice 0.939, while
the final epoch of the same run scored 0.885 (one case fell from 0.93 to 0.53). Here every CT is scored exactly once, by
a model that never trained on it and was never selected on it:
  * each fold trains for a fixed `epochs` (no validation during training);
  * the scored weights are the stochastic weight average (SWA) of the second half of training, with BatchNorm
    statistics recomputed on training patches afterwards, which is steadier than any single last epoch;
  * largest-connected-component post-processing, the same as figures_seg.
Writes reports/seg_cv_per_case.csv (one row per CT) and reports/seg_cv.json (mean, SD, median, bootstrap CI).
A finished fold is saved to models/seg_cv_fold{k}.pt and skipped on re-run, so a killed session resumes.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import json_hashing
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.networks.utils import one_hot
from torch.optim.swa_utils import AveragedModel

from medimg import ROOT, load_params, seed_everything
from medimg.seg.figures_seg import keep_largest
from medimg.seg.train_seg import build_model, transforms


def all_cases(sp: dict, seed: int) -> list[dict]:
    """Every labelled case once: MONAI's training + validation sections partition the 41 labelled volumes."""
    root = ROOT / sp["root_dir"]
    root.mkdir(parents=True, exist_ok=True)
    common = dict(root_dir=str(root), task=sp["task"], val_frac=0.2, seed=seed, cache_rate=0.0)
    first = DecathlonDataset(section="training", download=not (root / sp["task"]).exists(), **common).data
    cases = first + DecathlonDataset(section="validation", download=False, **common).data
    return sorted(cases, key=lambda c: c["image"])


def folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


@torch.no_grad()
def recompute_bn(model: torch.nn.Module, loader, device, batches: int) -> None:
    """Averaged weights need BatchNorm statistics of their own (torch.optim.swa_utils.update_bn, for dict batches)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over all batches below
    model.train()
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        model(batch["image"].to(device))
    model.eval()


def score(pred: torch.Tensor, y1: torch.Tensor, spacing) -> tuple[float, float]:
    p1 = one_hot(pred, 2)
    return (compute_dice(p1, y1, include_background=False).item(),
            compute_hausdorff_distance(p1, y1, include_background=False, percentile=95, spacing=spacing).item())


def run_fold(k: int, train_cases: list[dict], test_cases: list[dict], sp: dict, cv: dict, device) -> list[dict]:
    amp = sp["amp"] and device.type == "cuda"
    cache = ROOT / "data" / "processed" / "msd_cache"
    train_ds = PersistentDataset(train_cases, transforms(sp, True), cache_dir=cache / "train", hash_transform=json_hashing)
    test_ds = PersistentDataset(test_cases, transforms(sp, False), cache_dir=cache / "val", hash_transform=json_hashing)
    train_dl = DataLoader(train_ds, batch_size=sp["batch_size"], shuffle=True, num_workers=sp["num_workers"])

    model = build_model().to(device)
    swa = AveragedModel(model)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=sp["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cv["epochs"])
    scaler = torch.amp.GradScaler(enabled=amp)
    swa_start = int(cv["epochs"] * cv["swa_start_frac"])
    for epoch in range(cv["epochs"]):
        model.train()
        t0, total = time.time(), 0.0
        for batch in train_dl:
            x, y = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()
        if epoch >= swa_start:
            swa.update_parameters(model)
        if epoch % 25 == 0 or epoch == cv["epochs"] - 1:
            print(json.dumps({"fold": k, "epoch": epoch, "train_loss": round(total / len(train_dl), 4),
                              "epoch_sec": round(time.time() - t0, 1)}), flush=True)
    recompute_bn(swa.module, train_dl, device, batches=cv["bn_batches"])

    rows = []
    with torch.no_grad():
        for variant, net in (("swa", swa.module), ("final", model.eval())):
            for item, case in zip(test_ds, test_cases):
                x = item["image"][None].to(device).float()
                y1 = one_hot(item["label"][None].to(device).long(), 2)
                with torch.autocast(device.type, enabled=amp):
                    raw = sliding_window_inference(x, sp["roi_size"], 4, net, overlap=0.5).argmax(1, keepdim=True)
                pred = torch.from_numpy(keep_largest(raw[0, 0].cpu().numpy()))[None, None].to(device)
                (dice_raw, hd_raw), (dice, hd95) = score(raw, y1, sp["pixdim"]), score(pred, y1, sp["pixdim"])
                rows.append({"fold": k, "model": variant, "case": case["image"].split("/")[-1].split("\\")[-1].split(".")[0],
                             "dice": round(dice, 4), "hd95_mm": round(hd95, 2),
                             "dice_raw": round(dice_raw, 4), "hd95_raw_mm": round(hd_raw, 2)})
    torch.save({"state_dict": swa.module.state_dict(), "rows": rows, "test_cases": [c["image"] for c in test_cases]},
               ROOT / "models" / f"seg_cv_fold{k}.pt")
    return rows


def bootstrap_ci(x: np.ndarray, iters: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(iters)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def main() -> None:
    p = load_params()
    sp, cv = p["segmentation"], p["segmentation_cv"]
    seed_everything(p["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = all_cases(sp, p["seed"])
    (ROOT / "models").mkdir(exist_ok=True)
    rows = []
    for k, test_idx in enumerate(folds(len(cases), cv["folds"], p["seed"])):
        done = ROOT / "models" / f"seg_cv_fold{k}.pt"
        if done.exists():
            print(f"fold {k}: already done, reusing {done.name}")
            rows += torch.load(done, map_location="cpu", weights_only=False)["rows"]
            continue
        test = set(test_idx.tolist())
        rows += run_fold(k, [c for i, c in enumerate(cases) if i not in test], [cases[i] for i in sorted(test)],
                         sp, cv, device)

    df = pd.DataFrame(rows)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "seg_cv_per_case.csv", index=False)
    summary = {"n_cases": int(df[df.model == "swa"].shape[0]), "folds": cv["folds"], "epochs_per_fold": cv["epochs"],
               "selection": "none (fixed schedule; SWA of the second half, BN recomputed)",
               "postprocessing": "largest connected component"}
    for variant, g in df.groupby("model"):
        d = g.dice.to_numpy()
        summary[variant] = {"dice_mean": round(float(d.mean()), 4), "dice_sd": round(float(d.std(ddof=1)), 4),
                            "dice_median": round(float(np.median(d)), 4), "dice_min": round(float(d.min()), 4),
                            "dice_ci95": bootstrap_ci(d), "hd95_mm_median": round(float(g.hd95_mm.median()), 2),
                            "dice_raw_mean": round(float(g.dice_raw.mean()), 4),
                            "cases_below_0.9": int((d < 0.9).sum())}
    (out / "seg_cv.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
