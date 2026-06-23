"""
app/utils/quality_check.py
Image quality checks before running the pipeline.
Checks for blur, brightness, and contrast.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class QualityReport:
    passed: bool
    blur_score: float
    brightness_score: float
    contrast_score: float
    warnings: list[str]
    overall_score: float


def check_quality(image: Image.Image) -> QualityReport:
    """
    Run quality checks on a PIL image.
    Returns a QualityReport with scores and warnings.
    """
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    warnings = []

    # ── Blur check (Laplacian variance) ──
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_normalized = min(blur_score / 500.0, 1.0)
    if blur_score < 50:
        warnings.append("Image is too blurry. Please use a sharper image.")
    elif blur_score < 100:
        warnings.append("Image is slightly blurry. Results may be less accurate.")

    # ── Brightness check ──
    brightness_score = float(gray.mean())
    brightness_normalized = brightness_score / 255.0
    if brightness_score < 40:
        warnings.append("Image is too dark. Please use a brighter image.")
    elif brightness_score > 220:
        warnings.append("Image is too bright/overexposed. Results may be less accurate.")
    elif brightness_score < 60:
        warnings.append("Image is slightly dark. Consider adjusting brightness.")

    # ── Contrast check (std deviation) ──
    contrast_score = float(gray.std())
    contrast_normalized = min(contrast_score / 80.0, 1.0)
    if contrast_score < 20:
        warnings.append("Image has very low contrast. Results may be less accurate.")
    elif contrast_score < 35:
        warnings.append("Image has low contrast. Consider adjusting image settings.")

    # ── Overall score ──
    overall_score = round(
        (blur_normalized * 0.5 +
         (1 - abs(brightness_normalized - 0.5) * 2) * 0.3 +
         contrast_normalized * 0.2) * 100,
        1
    )

    passed = len([w for w in warnings if "too" in w]) == 0

    return QualityReport(
        passed=passed,
        blur_score=round(blur_score, 1),
        brightness_score=round(brightness_score, 1),
        contrast_score=round(contrast_score, 1),
        warnings=warnings,
        overall_score=overall_score,
    )