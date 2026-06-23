from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.api.predict import router as predict_router
from app.api.history import router as history_router
from app.core.dependencies import get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Leukemia Detection API",
        description="YOLOv8 + CNN blood-smear leukemia screening",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent.parent / "frontend" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates_dir = Path(__file__).parent.parent / "frontend" / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    app.include_router(predict_router)
    app.include_router(history_router)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @app.get("/history", response_class=HTMLResponse)
    async def history_page(request: Request):
        return templates.TemplateResponse(request, "history.html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    import yaml

    cfg_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    srv = cfg.get("server", {})
    uvicorn.run(
        "app.main:app",
        host=srv.get("host", "0.0.0.0"),
        port=srv.get("port", 8000),
        reload=True,
    )