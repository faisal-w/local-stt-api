"""Abstract base class for STT models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
import numpy as np


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    segments: list = field(default_factory=list)
    duration: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "segments": self.segments,
            "duration": self.duration,
        }


class STTModelBase(ABC):
    """Abstract base class for all STT model implementations."""
    
    def __init__(self, model_size: str = "base", device: str = "cpu", **kwargs):
        """Initialize the model.
        
        Args:
            model_size: Size/variant of the model to load.
            device: Device to run inference on ('cpu' or 'cuda').
            **kwargs: Additional model-specific parameters.
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass
    
    @property
    @abstractmethod
    def supported_sizes(self) -> list[str]:
        """Return list of supported model sizes."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory.
        
        Should set self._is_loaded = True after successful loading.
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory."""
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (mono, float32 or int16).
            sample_rate: Sample rate of the audio.
            language: Optional language code for transcription.
            
        Returns:
            TranscriptionResult containing the transcription.
        """
        pass
    
    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Transcribe streaming audio.
        
        Args:
            audio_chunks: Async iterator of audio chunks.
            sample_rate: Sample rate of the audio.
            language: Optional language code.
            
        Yields:
            TranscriptionResult for each processed chunk.
        """
        pass
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            "name": self.name,
            "model_size": self.model_size,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "supported_sizes": self.supported_sizes,
        }
