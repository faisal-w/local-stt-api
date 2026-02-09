"""Moonshine STT model implementation for low-latency transcription."""

import logging
from typing import AsyncIterator, Optional
import numpy as np

from app.models.base import STTModelBase, TranscriptionResult

logger = logging.getLogger(__name__)


class MoonshineModel(STTModelBase):
    """Moonshine implementation optimized for low-latency real-time transcription."""
    
    SIZES = ["tiny", "base"]
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize Moonshine model.
        
        Args:
            model_size: 'tiny' (27M params) or 'base' (61M params).
            device: 'cpu' or 'cuda'.
        """
        super().__init__(model_size, device, **kwargs)
        self._tokenizer = None
    
    @property
    def name(self) -> str:
        return "moonshine"
    
    @property
    def supported_sizes(self) -> list[str]:
        return self.SIZES
    
    def load_model(self) -> None:
        """Load the Moonshine model."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            # Try ONNX version first (faster, more portable)
            try:
                from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
                
                logger.info(f"Loading Moonshine ONNX model: {self.model_size}")
                
                self.model = MoonshineOnnxModel(model_name=self.model_size)
                self._tokenizer = load_tokenizer()
                self._use_onnx = True
                
            except ImportError:
                # Fall back to standard moonshine
                import moonshine
                
                logger.info(f"Loading Moonshine model: {self.model_size} on {self.device}")
                
                self.model = moonshine.load_model(self.model_size)
                self._tokenizer = moonshine.load_tokenizer()
                self._use_onnx = False
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            del self._tokenizer
            self.model = None
            self._tokenizer = None
            self._is_loaded = False
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("Model unloaded")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Moonshine.
        
        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate (should be 16kHz).
            language: Not used (Moonshine is English-only).
            
        Returns:
            TranscriptionResult with transcription text.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
        
        # Run inference
        if self._use_onnx:
            tokens = self.model.generate(audio)
            text = self._tokenizer.decode_batch(tokens)[0]
        else:
            import moonshine
            text = moonshine.transcribe(audio, self.model, self._tokenizer)
            if isinstance(text, list):
                text = text[0]
        
        return TranscriptionResult(
            text=text.strip(),
            language="en",  # Moonshine is English-only
            duration=len(audio) / 16000,
        )
    
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription optimized for low latency.
        
        Moonshine is optimized for short audio segments, making it ideal for
        real-time streaming with minimal latency.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        buffer = []
        buffer_duration = 0.0
        # Moonshine works well with shorter chunks (optimized for this)
        min_chunk_duration = 0.5  # 500ms minimum
        max_chunk_duration = 10.0  # 10 seconds max
        
        async for chunk in audio_chunks:
            if chunk.dtype == np.int16:
                chunk = chunk.astype(np.float32) / 32768.0
            
            buffer.append(chunk)
            buffer_duration += len(chunk) / sample_rate
            
            if buffer_duration >= min_chunk_duration:
                combined = np.concatenate(buffer)
                
                if sample_rate != 16000:
                    combined = self._resample(combined, sample_rate, 16000)
                
                # Transcribe
                if self._use_onnx:
                    tokens = self.model.generate(combined)
                    text = self._tokenizer.decode_batch(tokens)[0]
                else:
                    import moonshine
                    text = moonshine.transcribe(combined, self.model, self._tokenizer)
                    if isinstance(text, list):
                        text = text[0]
                
                if text.strip():
                    yield TranscriptionResult(
                        text=text.strip(),
                        language="en",
                    )
                
                if buffer_duration >= max_chunk_duration:
                    buffer = []
                    buffer_duration = 0.0
        
        # Process remaining
        if buffer:
            combined = np.concatenate(buffer)
            if sample_rate != 16000:
                combined = self._resample(combined, sample_rate, 16000)
            
            if self._use_onnx:
                tokens = self.model.generate(combined)
                text = self._tokenizer.decode_batch(tokens)[0]
            else:
                import moonshine
                text = moonshine.transcribe(combined, self.model, self._tokenizer)
                if isinstance(text, list):
                    text = text[0]
            
            if text.strip():
                yield TranscriptionResult(
                    text=text.strip(),
                    language="en",
                )
    
    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)
