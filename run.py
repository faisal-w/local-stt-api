"""
Local Speech-to-Text API
Run this script to start the FastAPI server.
"""

import uvicorn
from app.config import settings


def main():
    """Start the FastAPI server."""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
