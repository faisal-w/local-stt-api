# Local Speech-to-Text API

A FastAPI-based local speech-to-text service supporting multiple STT models with streaming capabilities.

## Supported Models

| Model | Description | VRAM | Best For |
|-------|-------------|------|----------|
| **faster-whisper** | CTranslate2-optimized Whisper | ~2-5GB | General accuracy + speed |
| **sensevoice** | FunASR SenseVoiceSmall | ~2GB | Multilingual + emotion |
| **moonshine** | Low-latency ASR | ~1GB | Real-time, CPU-friendly |

## Quick Start

### 1. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
python run.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/models` | List available models |
| `POST` | `/transcribe` | Transcribe uploaded audio file |
| `POST` | `/transcribe/stream` | Transcribe with SSE streaming |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8000/ws/transcribe` | Real-time audio streaming |

## Configuration

Set environment variables or create `.env` file:

```env
MODEL_NAME=faster-whisper    # faster-whisper, sensevoice, moonshine
MODEL_SIZE=base              # Model size variant
DEVICE=auto                  # cpu, cuda, auto
HOST=0.0.0.0
PORT=8000
```

## Usage Examples

### Transcribe Audio File

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={"model": "faster-whisper"}
    )
print(response.json())
```

### WebSocket Streaming

```python
import asyncio
import websockets

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/ws/transcribe") as ws:
        await ws.send('{"model": "faster-whisper"}')
        # Send audio chunks...
        async for message in ws:
            print(message)

asyncio.run(stream_audio())
```

## License

MIT License
