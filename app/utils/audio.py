"""Audio processing utilities."""

import io
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio from a file path.
    
    Args:
        file_path: Path to the audio file.
        
    Returns:
        Tuple of (audio_array, sample_rate).
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(file_path, dtype='float32')
    except Exception:
        # Fallback to librosa for more format support
        import librosa
        audio, sr = librosa.load(file_path, sr=None, mono=True)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    return audio.astype(np.float32), sr


def load_audio_from_bytes(
    audio_bytes: bytes,
    filename: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio from bytes.
    
    Args:
        audio_bytes: Raw audio file bytes.
        filename: Optional filename for format detection.
        
    Returns:
        Tuple of (audio_array, sample_rate).
    """
    buffer = io.BytesIO(audio_bytes)
    
    # Try soundfile first
    try:
        import soundfile as sf
        buffer.seek(0)
        audio, sr = sf.read(buffer, dtype='float32')
    except Exception as e:
        logger.debug(f"soundfile failed, trying librosa: {e}")
        # Fallback to librosa
        try:
            import librosa
            buffer.seek(0)
            audio, sr = librosa.load(buffer, sr=None, mono=True)
        except Exception as e2:
            raise ValueError(
                f"Could not load audio file. Supported formats: WAV, MP3, FLAC, OGG, M4A. "
                f"Error: {e2}"
            )
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    return audio.astype(np.float32), sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Simple linear interpolation fallback
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range.
    
    Args:
        audio: Audio array.
        
    Returns:
        Normalized audio array.
    """
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    
    # Already float, normalize if needed
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        return audio / max_val
    
    return audio.astype(np.float32)


def split_audio_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_sec: float = 5.0,
    overlap_sec: float = 0.5,
) -> list[np.ndarray]:
    """Split audio into overlapping chunks.
    
    Args:
        audio: Audio array.
        sample_rate: Sample rate.
        chunk_duration_sec: Duration of each chunk in seconds.
        overlap_sec: Overlap between chunks in seconds.
        
    Returns:
        List of audio chunks.
    """
    chunk_size = int(chunk_duration_sec * sample_rate)
    overlap_size = int(overlap_sec * sample_rate)
    step_size = chunk_size - overlap_size
    
    chunks = []
    for i in range(0, len(audio), step_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) > sample_rate * 0.1:  # At least 100ms
            chunks.append(chunk)
    
    return chunks
