"""
app/api/history.py
API endpoints for analysis history.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pathlib import Path

from app.core.history import get_all, delete_record, clear_all

router = APIRouter(prefix="/api/history", tags=["history"])

templates = Jinja2Templates(
    directory=str(Path(__file__).parent.parent.parent / "frontend" / "templates")
)


@router.get("/")
async def get_history():
    """Return all past analyses."""
    return {"history": get_all()}


@router.delete("/{record_id}")
async def delete_history_record(record_id: str):
    """Delete a single history record."""
    success = delete_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found.")
    return {"deleted": record_id}


@router.delete("/")
async def clear_history():
    """Clear all history."""
    clear_all()
    return {"status": "cleared"}


@router.get("/page", response_class=HTMLResponse)
async def history_page(request: Request):
    """Serve the history HTML page."""
    return templates.TemplateResponse(request, "history.html")