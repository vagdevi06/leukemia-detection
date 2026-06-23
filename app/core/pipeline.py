from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from app.models.yolo_detector import Detection, DetectionResult, YOLOCellDetector
from app.utils.image_utils import apply_clahe, load_image
from app.utils.gradcam import generate_gradcam_b64

AggregationMethod = Literal["majority", "weighted_avg", "max_conf"]


@dataclass
class CellPrediction:
    detection: Detection
    class_id: int
    class_name: str
    confidence: float
    probabilities: List[float] = field(default_factory=list)
    gradcam_b64: str = ""


@dataclass
class PipelineResult:
    slide_diagnosis: str
    slide_confidence: float
    leukemic_cell_ratio: float
    total_cells_detected: int
    leukemic_cells: int
    normal_cells: int
    cell_predictions: List[CellPrediction]
    inference_time_ms: float
    detection_result: DetectionResult
    error: Optional[str] = None


class LeukemiaDetectionPipeline:
    CLASS_NAMES = [
        "Normal Lymphocyte",
        "ALL Blast",
        "AML Blast",
        "CML Blast",
        "Suspicious Cell",
    ]
    LEUKEMIA_CLASS_IDS = [1, 2, 3, 4]  # all non-normal classes
    LEUKEMIA_CLASS_ID  = 1              # kept for Grad-CAM compat

    def __init__(self, detector, classifier, class_names=None,
                 max_cells=50, min_cell_conf=0.40, aggregation="majority",
                 batch_size=16, cnn_input_size=(224, 224),
                 normalize_mean=(0.485, 0.456, 0.406),
                 normalize_std=(0.229, 0.224, 0.225),
                 clahe=True, device="cpu"):
        self.detector = detector
        self.classifier = classifier
        self.class_names = class_names or self.CLASS_NAMES
        self.max_cells = max_cells
        self.min_cell_conf = min_cell_conf
        self.aggregation = aggregation
        self.batch_size = batch_size
        self.device = device
        self.clahe = clahe
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.cnn_input_size = cnn_input_size

        self.transform = T.Compose([
            T.Resize(cnn_input_size),
            T.ToTensor(),
            T.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
        ])

        if classifier is not None:
            self.classifier.eval()

    def run(self, image_input) -> PipelineResult:
        t0 = time.perf_counter()
        try:
            image = self._load(image_input)
            if self.clahe:
                image = apply_clahe(image)

            det_result = self.detector.detect(image)
            candidates = det_result.filter_by_conf(self.min_cell_conf).detections
            candidates = candidates[:self.max_cells]

            if not candidates:
                elapsed = (time.perf_counter() - t0) * 1000
                return PipelineResult(
                    slide_diagnosis="No Cells Detected",
                    slide_confidence=0.0,
                    leukemic_cell_ratio=0.0,
                    total_cells_detected=0,
                    leukemic_cells=0,
                    normal_cells=0,
                    cell_predictions=[],
                    inference_time_ms=elapsed,
                    detection_result=det_result,
                )

            cell_preds = self._classify_cells(image, candidates)
            diagnosis, confidence = self._aggregate(cell_preds)
            leukemic = [p for p in cell_preds if p.class_id in self.LEUKEMIA_CLASS_IDS]
            normal = [p for p in cell_preds if p.class_id not in self.LEUKEMIA_CLASS_IDS]

            elapsed = (time.perf_counter() - t0) * 1000
            return PipelineResult(
                slide_diagnosis=diagnosis,
                slide_confidence=confidence,
                leukemic_cell_ratio=len(leukemic) / max(len(cell_preds), 1),
                total_cells_detected=len(cell_preds),
                leukemic_cells=len(leukemic),
                normal_cells=len(normal),
                cell_predictions=cell_preds,
                inference_time_ms=elapsed,
                detection_result=det_result,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return PipelineResult(
                slide_diagnosis="Error",
                slide_confidence=0.0,
                leukemic_cell_ratio=0.0,
                total_cells_detected=0,
                leukemic_cells=0,
                normal_cells=0,
                cell_predictions=[],
                inference_time_ms=elapsed,
                detection_result=DetectionResult(),
                error=str(exc),
            )

    def _load(self, source) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (str, bytes)):
            return load_image(source)
        raise TypeError(f"Unsupported image source type: {type(source)}")

    @torch.no_grad()
    def _classify_cells(self, image, detections):
        crops = [det.crop(image) for det in detections]
        tensors = torch.stack([self.transform(c) for c in crops]).to(self.device)
        predictions = []

        for start in range(0, len(tensors), self.batch_size):
            batch = tensors[start: start + self.batch_size]
            if self.classifier is not None:
                probs = self.classifier.predict_proba(batch).cpu().numpy()
            else:
                rng = np.random.default_rng(start)
                probs = rng.dirichlet(np.ones(len(self.class_names)), size=len(batch))

            for i, prob_row in enumerate(probs):
                class_id = int(np.argmax(prob_row))

                # Generate Grad-CAM for leukemic cells
                gradcam_b64 = ""
                if class_id in self.LEUKEMIA_CLASS_IDS and self.classifier is not None:
                    try:
                        crop = crops[start + i]
                        gradcam_b64 = generate_gradcam_b64(
                            model=self.classifier,
                            cell_image=crop,
                            class_id=class_id,
                            device=self.device,
                        )
                    except Exception:
                        pass

                predictions.append(CellPrediction(
                    detection=detections[start + i],
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    confidence=float(prob_row[class_id]),
                    probabilities=prob_row.tolist(),
                    gradcam_b64=gradcam_b64,
                ))

        return predictions

    def _aggregate(self, preds):
        if not preds:
            return "No Cells Detected", 0.0

        leukemic_mask = np.array([
            p.class_id in self.LEUKEMIA_CLASS_IDS for p in preds
        ])

        if self.aggregation == "majority":
            is_leukemia = leukemic_mask.mean() >= 0.5
            conf = float(leukemic_mask.mean() if is_leukemia else 1 - leukemic_mask.mean())

        elif self.aggregation == "weighted_avg":
            weights = np.array([p.confidence for p in preds])
            weighted_ratio = float(np.dot(weights, leukemic_mask) / weights.sum())
            is_leukemia = weighted_ratio >= 0.5
            conf = weighted_ratio if is_leukemia else 1 - weighted_ratio

        else:
            leukemic_confs = [p.confidence for p in preds if p.class_id == self.LEUKEMIA_CLASS_ID]
            normal_confs = [p.confidence for p in preds if p.class_id != self.LEUKEMIA_CLASS_ID]
            if leukemic_confs and (not normal_confs or max(leukemic_confs) >= max(normal_confs)):
                is_leukemia, conf = True, max(leukemic_confs)
            else:
                is_leukemia, conf = False, max(normal_confs) if normal_confs else 0.0

        if is_leukemia:
            # Find most common leukemic subtype
            leukemic_preds = [p for p in preds if p.class_id in self.LEUKEMIA_CLASS_IDS]
            subtype_counts = {}
            for p in leukemic_preds:
                subtype_counts[p.class_name] = subtype_counts.get(p.class_name, 0) + 1
            dominant = max(subtype_counts, key=subtype_counts.get)
            label = f"Leukemia Detected — {dominant}"
        else:
            label = "No Leukemia Detected"
        return label, conf