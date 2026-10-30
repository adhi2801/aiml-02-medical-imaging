from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm


class CXRClassifier(nn.Module):
    """ImageNet-pretrained backbone + single-logit head (lung opacity vs. not).

    The forward pass can also return the last convolutional feature map, which is what
    Grad-CAM needs, so no fragile forward/backward hooks are required.
    """

    def __init__(self, arch: str = "densenet121", pretrained: bool = True):
        super().__init__()
        self.arch = arch
        if arch == "densenet121":
            m = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT if pretrained else None)
            # torchvision applies this ReLU functionally (and in place); make it an explicit module.
            self.features = nn.Sequential(m.features, nn.ReLU())
            n_feat = m.classifier.in_features
        elif arch == "efficientnet_b0":
            m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.features = m.features
            n_feat = m.classifier[1].in_features
        else:
            raise ValueError(f"unknown arch {arch!r}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(n_feat, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f = self.features(x)
        logit = self.head(self.dropout(torch.flatten(self.pool(f), 1))).squeeze(1)
        return (logit, f) if return_features else logit


def save_checkpoint(model: CXRClassifier, img_size: int, path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": model.arch, "img_size": img_size, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[CXRClassifier, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = CXRClassifier(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
