from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str = ""

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop(self.bbox)


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    def filter_by_conf(self, threshold: float) -> "DetectionResult":
        return DetectionResult(
            detections=[d for d in self.detections if d.confidence >= threshold],
            image_width=self.image_width,
            image_height=self.image_height,
        )


class YOLOCellDetector:
    def __init__(self, weights_path, img_size=640, conf_threshold=0.45,
                 iou_threshold=0.50, device="cpu"):
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
        self._mock_mode = False

        path = Path(weights_path)
        if path.exists():
            self._load_model(path)
        else:
            print(f"[YOLODetector] Weights not found. Running in MOCK mode.")
            self._mock_mode = True

    def _load_model(self, path):
        from ultralytics import YOLO
        self._model = YOLO(str(path))
        self._model.to(self.device)

    def detect(self, image: Image.Image) -> DetectionResult:
        w, h = image.size
        if self._mock_mode:
            return self._mock_detect(w, h)
        results = self._model.predict(
            source=image, imgsz=self.img_size,
            conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
        )
        return self._parse_results(results[0], w, h)

    @staticmethod
    def _parse_results(result, img_w, img_h) -> DetectionResult:
        detections = []
        names = result.names or {}
        if result.boxes is not None:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                detections.append(Detection(
                    x1=int(coords[0]), y1=int(coords[1]),
                    x2=int(coords[2]), y2=int(coords[3]),
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                    class_name=names.get(int(box.cls[0]), "cell"),
                ))
        return DetectionResult(detections=detections, image_width=img_w, image_height=img_h)

    @staticmethod
    def _mock_detect(w, h) -> DetectionResult:
        rng = np.random.default_rng(42)
        detections = []
        for _ in range(rng.integers(4, 12)):
            x1 = int(rng.integers(0, w - 80))
            y1 = int(rng.integers(0, h - 80))
            size = int(rng.integers(40, 100))
            detections.append(Detection(
                x1=x1, y1=y1,
                x2=min(x1 + size, w), y2=min(y1 + size, h),
                confidence=float(rng.uniform(0.45, 0.98)),
                class_id=rng.integers(0, 2),
                class_name="cell",
            ))
        return DetectionResult(detections=detections, image_width=w, image_height=h)