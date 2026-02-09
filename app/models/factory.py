"""Factory for creating STT model instances."""

import logging
from typing import Optional

from app.models.base import STTModelBase
from app.config import settings

logger = logging.getLogger(__name__)

# Registry of available models
_MODEL_REGISTRY: dict[str, type[STTModelBase]] = {}

# Cached model instance
_current_model: Optional[STTModelBase] = None


def _register_models():
    """Register all available models."""
    global _MODEL_REGISTRY
    
    if _MODEL_REGISTRY:
        return
    
    from app.models.faster_whisper import FasterWhisperModel
    from app.models.sensevoice import SenseVoiceModel
    from app.models.moonshine import MoonshineModel
    
    _MODEL_REGISTRY = {
        "faster-whisper": FasterWhisperModel,
        "sensevoice": SenseVoiceModel,
        "moonshine": MoonshineModel,
    }


def list_available_models() -> list[dict]:
    """List all available models with their info."""
    _register_models()
    
    models = []
    for name, model_class in _MODEL_REGISTRY.items():
        # Create temporary instance to get info
        temp = model_class()
        models.append({
            "name": name,
            "supported_sizes": temp.supported_sizes,
            "description": _get_model_description(name),
        })
    
    return models


def _get_model_description(name: str) -> str:
    """Get model description."""
    descriptions = {
        "faster-whisper": "CTranslate2-optimized Whisper. Best accuracy/speed balance.",
        "sensevoice": "FunASR SenseVoiceSmall. Multilingual with emotion detection.",
        "moonshine": "Low-latency model optimized for real-time transcription.",
    }
    return descriptions.get(name, "")


def get_model(
    model_name: Optional[str] = None,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    load: bool = True,
) -> STTModelBase:
    """Get or create a model instance.
    
    Args:
        model_name: Model name (faster-whisper, sensevoice, moonshine).
        model_size: Model size variant.
        device: Device to use (cpu, cuda, auto).
        compute_type: Compute type for faster-whisper.
        load: Whether to load the model immediately.
        
    Returns:
        STTModelBase instance.
    """
    global _current_model
    
    _register_models()
    
    # Use defaults from settings
    model_name = model_name or settings.MODEL_NAME
    model_size = model_size or settings.MODEL_SIZE
    device = device if device and device != "auto" else settings.get_device()
    compute_type = compute_type if compute_type and compute_type != "auto" else settings.get_compute_type()
    
    # Check if we can reuse current model
    if _current_model is not None:
        if (
            _current_model.name == model_name
            and _current_model.model_size == model_size
            and _current_model.device == device
        ):
            if load and not _current_model.is_loaded:
                _current_model.load_model()
            return _current_model
        else:
            # Unload current model before switching
            logger.info(f"Switching from {_current_model.name} to {model_name}")
            _current_model.unload_model()
    
    # Validate model name
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    
    # Create new model instance
    model_class = _MODEL_REGISTRY[model_name]
    
    kwargs = {
        "model_size": model_size,
        "device": device,
    }
    
    # Add compute_type for faster-whisper
    if model_name == "faster-whisper":
        kwargs["compute_type"] = compute_type
    
    _current_model = model_class(**kwargs)
    
    if load:
        _current_model.load_model()
    
    return _current_model


def get_current_model() -> Optional[STTModelBase]:
    """Get the currently loaded model, if any."""
    return _current_model


def unload_current_model() -> None:
    """Unload the current model."""
    global _current_model
    
    if _current_model is not None:
        _current_model.unload_model()
        _current_model = None
