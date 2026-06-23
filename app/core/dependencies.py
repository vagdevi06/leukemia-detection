from __future__ import annotations

import yaml
from functools import lru_cache
from pathlib import Path

from app.models.cnn_classifier import LeukemiaCNN
from app.models.yolo_detector import YOLOCellDetector
from app.core.pipeline import LeukemiaDetectionPipeline

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_pipeline() -> LeukemiaDetectionPipeline:
    cfg = load_config()

    yolo_cfg = cfg["model"]["yolo"]
    cnn_cfg  = cfg["model"]["cnn"]
    inf_cfg  = cfg["inference"]
    pre_cfg  = cfg["preprocessing"]

    detector = YOLOCellDetector(
        weights_path=yolo_cfg["weights"],
        img_size=yolo_cfg["img_size"],
        conf_threshold=yolo_cfg["conf_threshold"],
        iou_threshold=yolo_cfg["iou_threshold"],
        device=yolo_cfg["device"],
    )

    cnn_path = Path(cnn_cfg["weights"])
    if cnn_path.exists():
        classifier = LeukemiaCNN.load(str(cnn_path), device=cnn_cfg["device"])
    else:
        print(f"[Dependencies] CNN weights not found. Mock CNN active.")
        classifier = LeukemiaCNN(
            backbone=cnn_cfg["backbone"],
            num_classes=cnn_cfg["num_classes"],
        )
        classifier.eval()

    pipeline = LeukemiaDetectionPipeline(
        detector=detector,
        classifier=classifier,
        class_names=cfg["model"]["classes"],
        max_cells=inf_cfg["max_cells_per_image"],
        min_cell_conf=inf_cfg["min_cell_conf"],
        aggregation=inf_cfg["aggregation"],
        batch_size=inf_cfg["batch_size"],
        cnn_input_size=tuple(cnn_cfg["input_size"]),
        normalize_mean=tuple(pre_cfg["normalize_mean"]),
        normalize_std=tuple(pre_cfg["normalize_std"]),
        clahe=pre_cfg["clahe"],
        device=cnn_cfg["device"],
    )

    return pipeline