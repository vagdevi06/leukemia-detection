from __future__ import annotations

import uuid
import io
from typing import Dict
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.core.dependencies import get_pipeline, load_config
from app.core.pipeline import LeukemiaDetectionPipeline, PipelineResult
from app.utils.image_utils import annotate_image, image_to_base64, load_image, validate_image
from app.core.history import add_record
from app.utils.quality_check import check_quality

router = APIRouter(prefix="/api", tags=["detection"])

_result_store: Dict[str, dict] = {}


class PredictionResponse(BaseModel):
    request_id: str
    slide_diagnosis: str
    slide_confidence: float
    leukemic_cell_ratio: float
    total_cells_detected: int
    leukemic_cells: int
    normal_cells: int
    inference_time_ms: float
    annotated_image_b64: str
    cell_detail: list
    quality: dict | None = None
    error: str | None = None


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    pipeline: LeukemiaDetectionPipeline = Depends(get_pipeline),
):
    cfg = load_config()
    allowed = cfg["server"]["allowed_extensions"]
    max_mb  = cfg["server"]["max_upload_mb"]

    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Extension '{suffix}' not allowed. Accepted: {allowed}",
        )

    data = await file.read()
    try:
        validate_image(data, max_mb=max_mb)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    image = load_image(data)

    # Quality check
    quality = check_quality(image)
    quality_data = {
        "passed": quality.passed,
        "overall_score": quality.overall_score,
        "blur_score": quality.blur_score,
        "brightness_score": quality.brightness_score,
        "contrast_score": quality.contrast_score,
        "warnings": quality.warnings,
    }

    result: PipelineResult = pipeline.run(image)

    if result.error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error
        )

    annotated = annotate_image(image, result.cell_predictions)
    b64 = image_to_base64(annotated)

    cell_detail = [
        {
            "class_name": p.class_name,
            "confidence": round(p.confidence, 4),
            "bbox": list(p.detection.bbox),
            "probabilities": [round(x, 4) for x in p.probabilities],
            "gradcam_b64": p.gradcam_b64,
        }
        for p in result.cell_predictions
    ]

    request_id = str(uuid.uuid4())
    response = PredictionResponse(
        request_id=request_id,
        slide_diagnosis=result.slide_diagnosis,
        slide_confidence=round(result.slide_confidence, 4),
        leukemic_cell_ratio=round(result.leukemic_cell_ratio, 4),
        total_cells_detected=result.total_cells_detected,
        leukemic_cells=result.leukemic_cells,
        normal_cells=result.normal_cells,
        inference_time_ms=round(result.inference_time_ms, 1),
        annotated_image_b64=b64,
        cell_detail=cell_detail,
        quality=quality_data,
    )

    result_dict = response.model_dump()
    result_dict["image_name"] = file.filename or "unknown"
    _result_store[request_id] = result_dict
    add_record(result_dict)
    return response


@router.get("/results/{request_id}")
async def get_result(request_id: str):
    result = _result_store.get(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or expired.")
    return JSONResponse(content=result)


@router.get("/health")
async def health():
    return {"status": "ok", "service": "leukemia-detection"}


@router.post("/report")
async def generate_report(data: dict):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=20*mm, leftMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)

        styles = getSampleStyleSheet()
        elements = []

        title_style = ParagraphStyle('title',
            fontSize=22, fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1a1a2e'),
            spaceAfter=6, alignment=TA_CENTER)
        elements.append(Paragraph("LeukaScan Diagnostic Report", title_style))

        sub_style = ParagraphStyle('sub',
            fontSize=10, fontName='Helvetica',
            textColor=colors.HexColor('#666666'),
            spaceAfter=4, alignment=TA_CENTER)
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            sub_style))
        elements.append(Paragraph(
            f"Request ID: {data.get('request_id', 'N/A')}",
            sub_style))
        elements.append(Spacer(1, 8*mm))

        diagnosis = data.get('slide_diagnosis', 'N/A')
        is_positive = 'detected' in diagnosis.lower() and 'no' not in diagnosis.lower()
        diag_color = colors.HexColor('#c0392b') if is_positive else colors.HexColor('#27ae60')

        diag_style = ParagraphStyle('diag',
            fontSize=16, fontName='Helvetica-Bold',
            textColor=diag_color,
            spaceAfter=6, alignment=TA_CENTER)
        elements.append(Paragraph(f"Diagnosis: {diagnosis}", diag_style))
        elements.append(Spacer(1, 6*mm))

        summary_data = [
            ['Metric', 'Value'],
            ['Slide Confidence', f"{data.get('slide_confidence', 0)*100:.1f}%"],
            ['Blast Cell Ratio', f"{data.get('leukemic_cell_ratio', 0)*100:.1f}%"],
            ['Total Cells Detected', str(data.get('total_cells_detected', 0))],
            ['Leukemic Blasts', str(data.get('leukemic_cells', 0))],
            ['Normal Lymphocytes', str(data.get('normal_cells', 0))],
            ['Inference Time', f"{data.get('inference_time_ms', 0):.1f} ms"],
        ]

        summary_table = Table(summary_data, colWidths=[90*mm, 80*mm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 11),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1),
             [colors.HexColor('#f8f9fa'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 8*mm))

        cell_detail = data.get('cell_detail', [])
        if cell_detail:
            heading_style = ParagraphStyle('heading',
                fontSize=13, fontName='Helvetica-Bold',
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=4)
            elements.append(Paragraph("Per-Cell Breakdown", heading_style))
            elements.append(Spacer(1, 3*mm))

            cell_data = [['#', 'Classification', 'Confidence', 'Blast Probability']]
            for i, cell in enumerate(cell_detail):
                blast_prob = cell.get('probabilities', [0, 0])
                blast_prob_val = blast_prob[1] if len(blast_prob) > 1 else 0
                cell_data.append([
                    str(i + 1),
                    cell.get('class_name', 'N/A'),
                    f"{cell.get('confidence', 0)*100:.1f}%",
                    f"{blast_prob_val*100:.1f}%",
                ])

            cell_table = Table(cell_data, colWidths=[15*mm, 70*mm, 45*mm, 45*mm])
            cell_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',   (0,0), (-1,-1), 9),
                ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
                ('ROWBACKGROUNDS', (0,1), (-1,-1),
                 [colors.HexColor('#f8f9fa'), colors.white]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
            ]))
            elements.append(cell_table)
            elements.append(Spacer(1, 8*mm))

        disclaimer_style = ParagraphStyle('disclaimer',
            fontSize=8, fontName='Helvetica',
            textColor=colors.HexColor('#888888'),
            alignment=TA_CENTER)
        elements.append(Paragraph(
            "This report is a research aid only and is not a certified medical device. "
            "All results must be reviewed by a qualified haematologist before clinical use.",
            disclaimer_style))

        doc.build(elements)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=leukascan_report.pdf"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}"
        )