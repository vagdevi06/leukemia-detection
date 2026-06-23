from __future__ import annotations

import io
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_image(source) -> Image.Image:
    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    if isinstance(source, str) and source.startswith("data:image"):
        b64 = source.split(",", 1)[1]
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise TypeError(f"Cannot load image from {type(source)}")


def apply_clahe(image: Image.Image, clip_limit: float = 2.0,
                tile_grid=(8, 8)) -> Image.Image:
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    merged = cv2.merge([cl, a, b])
    bgr_out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)


def annotate_image(image: Image.Image, cell_predictions,
                   class_colors=None, line_width=2, font_size=12) -> Image.Image:
    if class_colors is None:
        class_colors = {
            0: (72, 199, 142),
            1: (255, 92, 92),
        }
    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for pred in cell_predictions:
        color = class_colors.get(pred.class_id, (200, 200, 200))
        x1, y1, x2, y2 = pred.detection.bbox
        draw.rectangle([x1, y1, x2, y2],
                       fill=(*color, 40), outline=(*color, 220), width=line_width)
        label = f"{pred.class_name} {pred.confidence:.0%}"
        draw.text((x1 + 3, y1 + 2), label, fill=(*color, 255), font=font)

    combined = Image.alpha_composite(annotated, overlay)
    return combined.convert("RGB")


def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def validate_image(data: bytes, max_mb: float = 20.0) -> None:
    max_bytes = int(max_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise ValueError(f"Image exceeds {max_mb} MB limit.")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except Exception as exc:
        raise ValueError(f"Invalid image file: {exc}") from exc