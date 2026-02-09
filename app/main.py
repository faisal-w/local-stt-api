"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import router, websocket_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Local Speech-to-Text API...")
    logger.info(f"Default model: {settings.MODEL_NAME}")
    logger.info(f"Device: {settings.get_device()}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")
    from app.models.factory import unload_current_model
    unload_current_model()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Local Speech-to-Text API",
    description="Local STT service supporting multiple models with streaming capabilities.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)
app.include_router(websocket_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.models.factory import get_current_model
    
    model = get_current_model()
    
    return {
        "status": "healthy",
        "model_loaded": model.name if model and model.is_loaded else None,
        "device": settings.get_device(),
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Local Speech-to-Text API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
    }
