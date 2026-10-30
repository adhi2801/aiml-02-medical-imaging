"""DVC stage `train_cls`: transfer-learn a chest X-ray classifier (lung opacity vs. not).

Logs to Weights & Biases when WANDB_API_KEY is set (otherwise runs offline-free).
Writes models/classifier.pt (best val AUROC) and reports/train_metrics.json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from medimg import ROOT, load_params, seed_everything
from medimg.data.dataset import RSNADataset
from medimg.models.classifier import CXRClassifier, save_checkpoint


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y, _ in loader:
        with torch.autocast(device.type, enabled=amp):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def main() -> None:
    p = load_params()
    cp = p["classifier"]
    seed_everything(p["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cp["amp"] and device.type == "cuda"

    data_dir = ROOT / p["data"]["processed_dir"]
    meta = pd.read_csv(data_dir / "metadata.csv")
    train_df, val_df = meta[meta.split == "train"], meta[meta.split == "val"]

    def loader(df, train):
        return DataLoader(RSNADataset(df, data_dir / "images", cp["img_size"], train),
                          batch_size=cp["batch_size"], shuffle=train, num_workers=cp["num_workers"],
                          pin_memory=device.type == "cuda", drop_last=train, persistent_workers=cp["num_workers"] > 0)

    train_dl, val_dl = loader(train_df, True), loader(val_df, False)

    model = CXRClassifier(cp["arch"]).to(device)
    pos_weight = torch.tensor((train_df.label == 0).sum() / (train_df.label == 1).sum(), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cp["lr"], weight_decay=cp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cp["lr"], epochs=cp["epochs"],
                                                    steps_per_epoch=len(train_dl), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=amp)

    run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(project="aiml02-cxr", config={**cp, "seed": p["seed"], "n_train": len(train_df)})

    best_auc, best_epoch, history = -1.0, -1, []
    ckpt_path = ROOT / "models" / "classifier.pt"
    for epoch in range(cp["epochs"]):
        model.train()
        t0, running = time.time(), 0.0
        for x, y, _ in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        val_logits, val_y = predict(model, val_dl, device, amp)
        val_auc = roc_auc_score(val_y, val_logits)
        rec = {"epoch": epoch, "train_loss": running / len(train_dl), "val_auroc": val_auc,
               "lr": scheduler.get_last_lr()[0], "epoch_sec": time.time() - t0}
        history.append(rec)
        print(json.dumps(rec))
        if run:
            run.log(rec)

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            save_checkpoint(model, cp["img_size"], ckpt_path, val_auroc=val_auc, epoch=epoch)
        elif epoch - best_epoch >= cp["early_stop_patience"]:
            print(f"early stop: no val AUROC gain for {cp['early_stop_patience']} epochs")
            break

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "train_metrics.json").write_text(json.dumps(
        {"best_val_auroc": round(best_auc, 4), "best_epoch": best_epoch, "epochs_run": len(history)}, indent=2))
    pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
    if run:
        run.summary["best_val_auroc"] = best_auc
        run.finish()


if __name__ == "__main__":
    main()
