"""
app/core/history.py
Simple JSON-based history storage for past analyses.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path("data/history.json")


def _load() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save(records: list) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, indent=2)


def add_record(result: dict) -> dict:
    records = _load()
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "slide_diagnosis": result.get("slide_diagnosis"),
        "slide_confidence": result.get("slide_confidence"),
        "leukemic_cell_ratio": result.get("leukemic_cell_ratio"),
        "total_cells_detected": result.get("total_cells_detected"),
        "leukemic_cells": result.get("leukemic_cells"),
        "normal_cells": result.get("normal_cells"),
        "inference_time_ms": result.get("inference_time_ms"),
        "request_id": result.get("request_id"),
        "image_name": result.get("image_name", "unknown"),
    }
    records.insert(0, record)
    records = records[:50]  # keep last 50
    _save(records)
    return record


def get_all() -> list:
    return _load()


def delete_record(record_id: str) -> bool:
    records = _load()
    new_records = [r for r in records if r["id"] != record_id]
    if len(new_records) == len(records):
        return False
    _save(new_records)
    return True


def clear_all() -> None:
    _save([])