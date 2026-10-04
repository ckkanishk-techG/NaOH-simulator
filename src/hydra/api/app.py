"""FastAPI application: REST + static frontend, served entirely from local files."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from ..config import Settings

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
DISCLAIMER = (
    "HYDRA is a predictive research model, not a safety certification. Physical experiments "
    "need supervision, venting, pressure relief and protective equipment independent of software."
)


class Health(BaseModel):
    status: str
    version: str
    mode: str
    telemetry: bool
    offline_enforced: bool


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="HYDRA", version=__version__, docs_url=None, redoc_url=None)
    app.state.settings = settings

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        return Health(
            status="ok",
            version=__version__,
            mode=settings.mode,
            telemetry=False,
            offline_enforced=settings.enforce_offline,
        )

    @app.get("/api/disclaimer")
    def disclaimer() -> JSONResponse:
        return JSONResponse({"text": DISCLAIMER})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
