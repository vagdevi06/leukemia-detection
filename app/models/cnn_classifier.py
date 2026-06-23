from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal

BackboneName = Literal["resnet50", "efficientnet_b0", "mobilenet_v3"]


class LeukemiaCNN(nn.Module):
    def __init__(self, backbone: BackboneName = "resnet50",
                 num_classes: int = 2, dropout: float = 0.4,
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.backbone, in_features = self._build_backbone(backbone, pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def _build_backbone(self, name, pretrained):
        weights_arg = "DEFAULT" if pretrained else None

        if name == "resnet50":
            net = models.resnet50(weights=weights_arg)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
            return net, in_features

        if name == "efficientnet_b0":
            net = models.efficientnet_b0(weights=weights_arg)
            in_features = net.classifier[1].in_features
            net.classifier = nn.Identity()
            return net, in_features

        if name == "mobilenet_v3":
            net = models.mobilenet_v3_large(weights=weights_arg)
            in_features = net.classifier[3].in_features
            net.classifier = nn.Identity()
            return net, in_features

        raise ValueError(f"Unknown backbone: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    @classmethod
    def load(cls, weights_path: str, device: str = "cpu") -> "LeukemiaCNN":
        ckpt = torch.load(weights_path, map_location=device)
        cfg = ckpt.get("config", {})
        model = cls(
            backbone=cfg.get("backbone", "resnet50"),
            num_classes=cfg.get("num_classes", 2),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model.to(device)

    def save(self, path: str) -> None:
        torch.save({
            "model": self.state_dict(),
            "config": {
                "backbone": self.backbone_name,
                "num_classes": self.num_classes,
            },
        }, path)