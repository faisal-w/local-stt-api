"""Faster-Whisper STT model implementation using CTranslate2."""

import logging
from typing import AsyncIterator, Optional
import numpy as np

from app.models.base import STTModelBase, TranscriptionResult

logger = logging.getLogger(__name__)


class FasterWhisperModel(STTModelBase):
    """Faster-Whisper implementation using CTranslate2 backend."""
    
    SIZES = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
             "medium", "medium.en", "large-v2", "large-v3"]
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        **kwargs,
    ):
        """Initialize Faster-Whisper model.
        
        Args:
            model_size: One of tiny, base, small, medium, large-v2, large-v3.
            device: 'cpu' or 'cuda'.
            compute_type: 'int8', 'float16', or 'float32'.
        """
        super().__init__(model_size, device, **kwargs)
        self.compute_type = compute_type
        self._vad_model = None
    
    @property
    def name(self) -> str:
        return "faster-whisper"
    
    @property
    def supported_sizes(self) -> list[str]:
        return self.SIZES
    
    def load_model(self) -> None:
        """Load the Faster-Whisper model."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            from faster_whisper import WhisperModel
            
            logger.info(
                f"Loading faster-whisper model: {self.model_size} "
                f"on {self.device} with {self.compute_type}"
            )
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
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
            
            # Clear CUDA cache if available
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
        """Transcribe audio using Faster-Whisper.
        
        Args:
            audio: Audio data (mono, float32 normalized to [-1, 1] or int16).
            sample_rate: Sample rate (will be resampled to 16kHz if different).
            language: Optional language code (e.g., 'en', 'zh').
            
        Returns:
            TranscriptionResult with transcription text and metadata.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure audio is float32 normalized
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
        
        # Transcribe
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        # Collect segments
        segment_list = []
        full_text = []
        
        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })
            full_text.append(segment.text.strip())
        
        return TranscriptionResult(
            text=" ".join(full_text),
            language=info.language,
            confidence=info.language_probability,
            segments=segment_list,
            duration=info.duration,
        )
    
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription for real-time audio.
        
        Accumulates audio in chunks and transcribes when VAD detects speech end.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        buffer = []
        buffer_duration = 0.0
        min_chunk_duration = 1.0  # Minimum 1 second of audio before processing
        max_chunk_duration = 30.0  # Maximum 30 seconds
        
        async for chunk in audio_chunks:
            # Normalize chunk
            if chunk.dtype == np.int16:
                chunk = chunk.astype(np.float32) / 32768.0
            
            buffer.append(chunk)
            buffer_duration += len(chunk) / sample_rate
            
            # Process when we have enough audio
            if buffer_duration >= min_chunk_duration:
                combined = np.concatenate(buffer)
                
                # Resample if needed
                if sample_rate != 16000:
                    combined = self._resample(combined, sample_rate, 16000)
                
                # Transcribe
                segments, info = self.model.transcribe(
                    combined,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
                
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())
                
                if text_parts:
                    yield TranscriptionResult(
                        text=" ".join(text_parts),
                        language=info.language,
                        confidence=info.language_probability,
                    )
                
                # Reset buffer if we've hit max duration
                if buffer_duration >= max_chunk_duration:
                    buffer = []
                    buffer_duration = 0.0
        
        # Process remaining audio
        if buffer:
            combined = np.concatenate(buffer)
            if sample_rate != 16000:
                combined = self._resample(combined, sample_rate, 16000)
            
            segments, info = self.model.transcribe(
                combined,
                language=language,
                beam_size=5,
            )
            
            text_parts = [seg.text.strip() for seg in segments]
            if text_parts:
                yield TranscriptionResult(
                    text=" ".join(text_parts),
                    language=info.language,
                    confidence=info.language_probability,
                )
    
    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation fallback
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)
