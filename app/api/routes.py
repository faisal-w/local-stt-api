"""REST API routes for speech-to-text transcription."""

import logging
from typing import Optional
import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models import get_model, list_available_models, TranscriptionResult
from app.utils.audio import load_audio_from_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


class TranscribeRequest(BaseModel):
    """Request model for transcription."""
    model: Optional[str] = None
    model_size: Optional[str] = None
    language: Optional[str] = None


class TranscribeResponse(BaseModel):
    """Response model for transcription."""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    segments: list = []


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    supported_sizes: list[str]
    description: str


class ModelStatus(BaseModel):
    """Current model status."""
    name: str
    model_size: str
    device: str
    is_loaded: bool
    supported_sizes: list[str]


@router.get("/models", response_model=list[ModelInfo])
async def get_available_models():
    """List all available STT models."""
    return list_available_models()


@router.get("/models/current", response_model=Optional[ModelStatus])
async def get_current_model_status():
    """Get the currently loaded model status."""
    from app.models.factory import get_current_model
    
    model = get_current_model()
    if model is None:
        return None
    
    return model.get_info()


@router.post("/models/{model_name}/load", response_model=ModelStatus)
async def load_model(
    model_name: str,
    model_size: Optional[str] = None,
):
    """Load a specific model.
    
    Args:
        model_name: Name of the model to load.
        model_size: Optional size variant.
    """
    try:
        model = get_model(
            model_name=model_name,
            model_size=model_size,
            load=True,
        )
        return model.get_info()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    model_size: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    """Transcribe an uploaded audio file.
    
    Supports WAV, MP3, M4A, FLAC, OGG, and other common formats.
    
    Args:
        file: Audio file to transcribe.
        model: Model to use (faster-whisper, sensevoice, moonshine).
        model_size: Model size variant.
        language: Language hint for transcription.
        
    Returns:
        Transcription result with text and metadata.
    """
    try:
        # Read file content
        content = await file.read()
        
        # Load audio
        audio, sample_rate = load_audio_from_bytes(content, file.filename)
        
        # Get model
        stt_model = get_model(
            model_name=model,
            model_size=model_size,
            load=True,
        )
        
        # Transcribe
        result = stt_model.transcribe(audio, sample_rate, language)
        
        return TranscribeResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            duration=result.duration,
            segments=result.segments,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@router.post("/transcribe/stream")
async def transcribe_audio_stream(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    model_size: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    """Transcribe audio with Server-Sent Events streaming.
    
    Returns transcription results as SSE events as they become available.
    """
    try:
        content = await file.read()
        audio, sample_rate = load_audio_from_bytes(content, file.filename)
        
        stt_model = get_model(
            model_name=model,
            model_size=model_size,
            load=True,
        )
        
        async def generate_sse():
            """Generate SSE events from transcription."""
            import json
            
            # For file upload, we process in chunks to simulate streaming
            chunk_size = int(sample_rate * 5)  # 5-second chunks
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                result = stt_model.transcribe(chunk, sample_rate, language)
                
                if result.text:
                    event_data = json.dumps({
                        "text": result.text,
                        "language": result.language,
                        "is_final": (i + chunk_size >= len(audio)),
                    })
                    yield f"data: {event_data}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Streaming transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming transcription failed: {e}")
