"""WebSocket handler for real-time audio streaming transcription."""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np

from app.models import get_model

logger = logging.getLogger(__name__)

websocket_router = APIRouter()


class AudioStreamProcessor:
    """Process audio stream and yield transcriptions."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        self.model_name = model_name
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate
        self.model = None
        self.buffer = []
        self.buffer_duration = 0.0
        
    async def initialize(self):
        """Initialize the model."""
        self.model = get_model(
            model_name=self.model_name,
            model_size=self.model_size,
            load=True,
        )
    
    async def process_chunk(self, audio_bytes: bytes) -> Optional[dict]:
        """Process an audio chunk and return transcription if ready.
        
        Args:
            audio_bytes: Raw PCM audio bytes (16-bit signed, mono).
            
        Returns:
            Transcription result dict or None if buffer not ready.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Convert bytes to numpy array (assuming 16-bit signed PCM)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio.astype(np.float32) / 32768.0
        
        self.buffer.append(audio_float)
        self.buffer_duration += len(audio_float) / self.sample_rate
        
        # Process when we have enough audio (5 seconds for full sentences)
        if self.buffer_duration >= 5.0:
            combined = np.concatenate(self.buffer)

            result = self.model.transcribe(
                combined,
                self.sample_rate,
                self.language,
            )

            # Clear buffer after transcription to avoid duplicates
            self.buffer = []
            self.buffer_duration = 0.0

            if result.text:
                return {
                    "type": "transcription",
                    "text": result.text,
                    "language": result.language,
                    "is_partial": True,
                }
        
        return None
    
    async def finalize(self) -> Optional[dict]:
        """Process remaining audio in buffer."""
        if not self.buffer or self.model is None:
            return None
        
        combined = np.concatenate(self.buffer)
        result = self.model.transcribe(
            combined,
            self.sample_rate,
            self.language,
        )
        
        self.buffer = []
        self.buffer_duration = 0.0
        
        if result.text:
            return {
                "type": "transcription",
                "text": result.text,
                "language": result.language,
                "is_partial": False,
            }
        
        return None


@websocket_router.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for real-time audio transcription.
    
    Protocol:
    1. Client connects
    2. Client sends JSON config: {"model": "faster-whisper", "sample_rate": 16000}
    3. Client streams raw PCM audio bytes (16-bit signed, mono)
    4. Server sends JSON transcriptions: {"type": "transcription", "text": "...", "is_partial": true/false}
    5. Client sends "END" to signal end of audio
    6. Server sends final transcription and closes
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    processor = None
    
    try:
        # Receive configuration
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        
        logger.info(f"WebSocket config: {config}")
        
        # Initialize processor
        processor = AudioStreamProcessor(
            model_name=config.get("model"),
            model_size=config.get("model_size"),
            language=config.get("language"),
            sample_rate=config.get("sample_rate", 16000),
        )
        
        await processor.initialize()
        
        # Send ready signal
        await websocket.send_json({
            "type": "ready",
            "model": processor.model.name if processor.model else None,
        })
        
        # Process audio stream
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0,  # 30 second timeout
                )
                
                if message["type"] == "websocket.disconnect":
                    break
                
                if "text" in message:
                    text = message["text"]
                    if text == "END":
                        # Finalize and send remaining transcription
                        final_result = await processor.finalize()
                        if final_result:
                            await websocket.send_json(final_result)
                        
                        await websocket.send_json({"type": "done"})
                        break
                
                elif "bytes" in message:
                    audio_bytes = message["bytes"]
                    result = await processor.process_chunk(audio_bytes)
                    
                    if result:
                        await websocket.send_json(result)
                        
            except asyncio.TimeoutError:
                logger.warning("WebSocket timeout, closing connection")
                await websocket.send_json({
                    "type": "error",
                    "message": "Connection timeout",
                })
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON received: {e}")
        await websocket.send_json({
            "type": "error",
            "message": "Invalid JSON configuration",
        })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except:
            pass
    finally:
        logger.info("WebSocket connection closed")
