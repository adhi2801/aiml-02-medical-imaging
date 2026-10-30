"""Grad-CAM (Selvaraju et al., 2017) plus metrics that check whether the heatmap is *right*.

A heatmap that looks plausible is not evidence. RSNA ships radiologist bounding boxes, so we
measure localization quantitatively:
  * pointing game - is the single hottest pixel inside a radiologist box?
  * IoU           - overlap between the thresholded CAM region and the union of boxes.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (B,3,H,W). Returns (logits (B,), cams (B,H,W) in [0,1])."""
    model.eval()
    with torch.enable_grad():
        logit, feats = model(x, return_features=True)
        # Samples are independent in eval mode, so d(sum)/d(feats) gives per-sample gradients.
        grads = torch.autograd.grad(logit.sum(), feats)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    flat = cam.flatten(1)
    lo, hi = flat.min(1).values[:, None, None], flat.max(1).values[:, None, None]
    cam = (cam - lo) / (hi - lo).clamp_min(1e-8)
    return logit.detach(), cam.detach()


def boxes_to_mask(boxes: list[list[float]], orig_hw: tuple[int, int], out_hw: tuple[int, int]) -> np.ndarray:
    """Boxes are [x, y, w, h] in original pixel coords; returns a bool mask at out_hw."""
    mask = np.zeros(out_hw, dtype=bool)
    sy, sx = out_hw[0] / orig_hw[0], out_hw[1] / orig_hw[1]
    for x, y, w, h in boxes:
        mask[int(y * sy):int(np.ceil((y + h) * sy)), int(x * sx):int(np.ceil((x + w) * sx))] = True
    return mask


def pointing_hit(cam: np.ndarray, gt_mask: np.ndarray) -> bool:
    iy, ix = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(gt_mask[iy, ix])


def cam_iou(cam: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    pred = cam >= threshold
    union = np.logical_or(pred, gt_mask).sum()
    return float(np.logical_and(pred, gt_mask).sum() / union) if union else 0.0


def overlay(gray_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Return an RGB uint8 image with the CAM blended over the scan."""
    cam = cv2.resize(cam.astype(np.float32), gray_u8.shape[::-1])
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(cv2.addWeighted(heat, alpha, base, 1 - alpha, 0), cv2.COLOR_BGR2RGB)


def cam_to_rgba(cam: np.ndarray, out_hw: tuple[int, int], min_alpha_at: float = 0.2) -> np.ndarray:
    """Transparent heatmap PNG for the web viewer: cold regions fully transparent."""
    cam = cv2.resize(cam.astype(np.float32), out_hw[::-1])
    rgb = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    a = np.clip((cam - min_alpha_at) / (1 - min_alpha_at), 0, 1)
    return np.dstack([rgb, (a * 255).astype(np.uint8)])
