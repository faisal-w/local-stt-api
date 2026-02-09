"""SenseVoice STT model implementation using FunASR."""

import logging
from typing import AsyncIterator, Optional
import numpy as np

from app.models.base import STTModelBase, TranscriptionResult

logger = logging.getLogger(__name__)


class SenseVoiceModel(STTModelBase):
    """SenseVoiceSmall implementation using FunASR toolkit."""
    
    SIZES = ["small"]  # SenseVoice currently only has small variant
    MODEL_ID = "iic/SenseVoiceSmall"
    
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize SenseVoice model.
        
        Args:
            model_size: Model size (currently only 'small' is supported).
            device: 'cpu' or 'cuda'.
        """
        super().__init__(model_size, device, **kwargs)
    
    @property
    def name(self) -> str:
        return "sensevoice"
    
    @property
    def supported_sizes(self) -> list[str]:
        return self.SIZES
    
    def load_model(self) -> None:
        """Load the SenseVoice model."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            from funasr import AutoModel
            
            logger.info(f"Loading SenseVoice model on {self.device}")
            
            # Map device to FunASR format
            device_str = "cuda:0" if self.device == "cuda" else "cpu"
            
            self.model = AutoModel(
                model=self.MODEL_ID,
                trust_remote_code=True,
                device=device_str,
            )
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
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
        """Transcribe audio using SenseVoice.
        
        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            language: Optional language hint (auto-detected if not provided).
            
        Returns:
            TranscriptionResult with transcription and optional emotion/event info.
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
        result = self.model.generate(
            input=audio,
            cache={},
            language=language or "auto",
            use_itn=True,  # Inverse text normalization
        )
        
        # Parse result
        if result and len(result) > 0:
            output = result[0]
            
            # SenseVoice returns text with possible emotion/event tags
            text = output.get("text", "")
            
            # Clean up special tokens if present
            text = self._clean_text(text)
            
            return TranscriptionResult(
                text=text,
                language=output.get("language", language),
                confidence=output.get("confidence"),
                duration=len(audio) / 16000,
            )
        
        return TranscriptionResult(text="", duration=len(audio) / 16000)
    
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription for real-time audio."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        buffer = []
        buffer_duration = 0.0
        min_chunk_duration = 2.0  # SenseVoice works better with slightly longer chunks
        max_chunk_duration = 30.0
        
        async for chunk in audio_chunks:
            if chunk.dtype == np.int16:
                chunk = chunk.astype(np.float32) / 32768.0
            
            buffer.append(chunk)
            buffer_duration += len(chunk) / sample_rate
            
            if buffer_duration >= min_chunk_duration:
                combined = np.concatenate(buffer)
                
                if sample_rate != 16000:
                    combined = self._resample(combined, sample_rate, 16000)
                
                result = self.model.generate(
                    input=combined,
                    cache={},
                    language=language or "auto",
                    use_itn=True,
                )
                
                if result and len(result) > 0:
                    text = self._clean_text(result[0].get("text", ""))
                    if text:
                        yield TranscriptionResult(
                            text=text,
                            language=result[0].get("language"),
                        )
                
                if buffer_duration >= max_chunk_duration:
                    buffer = []
                    buffer_duration = 0.0
        
        # Process remaining
        if buffer:
            combined = np.concatenate(buffer)
            if sample_rate != 16000:
                combined = self._resample(combined, sample_rate, 16000)
            
            result = self.model.generate(
                input=combined,
                cache={},
                language=language or "auto",
                use_itn=True,
            )
            
            if result and len(result) > 0:
                text = self._clean_text(result[0].get("text", ""))
                if text:
                    yield TranscriptionResult(text=text)
    
    def _clean_text(self, text: str) -> str:
        """Remove special tokens from SenseVoice output."""
        # Remove emotion and event tags like <|HAPPY|>, <|BGM|>, etc.
        import re
        text = re.sub(r"<\|[^|]+\|>", "", text)
        return text.strip()
    
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
