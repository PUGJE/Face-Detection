"""
FastAPI Application — Main Entry Point

Responsibilities of this file are intentionally minimal:
  1. Create the FastAPI app with metadata.
  2. Register the CORS middleware.
  3. Mount static files (production build only).
  4. Wire up the database during the lifespan context.
  5. Include all route modules.

Business logic lives in backend/api/routes/ and backend/services/.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database.connection import init_database
from backend.config import settings
from backend.api.routes import health, students, faces, attendance

# ---------------------------------------------------------------------------
# Logging — configured once at the application entry point
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup & shutdown logic
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before the server accepts requests."""
    logger.info("Starting up …")
    settings.create_directories()   # ensure data dirs exist
    init_database()                  # create tables if needed
    logger.info(f"Listening at http://{settings.host}:{settings.port}")
    logger.info(f"API docs at  http://{settings.host}:{settings.port}/docs")
    yield
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Face Recognition Attendance System",
    description="Smart attendance system with face recognition and anti-spoofing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow all origins in development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve compiled Next.js static assets (production only)
_static_dir = Path(__file__).parent / "frontend" / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
    logger.info(f"Static files mounted from {_static_dir}")

# ---------------------------------------------------------------------------
# Route modules
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(students.router)
app.include_router(faces.router)
app.include_router(attendance.router)


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )
