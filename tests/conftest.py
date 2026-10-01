import base64
import io
import json
import wave

import httpx
import pytest


@pytest.fixture
def sse_audio():
    def make(pcm=None, *, duration=1.0, finish=True):
        if pcm is None:
            pcm = b"\0\0" * int(duration * 24000)
        output = {"output": {"audio": {"data": base64.b64encode(pcm).decode()}}}
        data = f"data: {json.dumps(output)}\n\n"
        if finish:
            data += 'data: {"output": {"finish_reason": "stop"}}\n\n'
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=data)

    return make


@pytest.fixture
def tts_endpoint():
    return (
        "https://test-workspace.cn-beijing.maas.aliyuncs.com"
        "/api/v1/services/audio/tts/SpeechSynthesizer"
    )


@pytest.fixture
def wav_bytes():
    def make(duration=1.0, rate=24000):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(b"\0\0" * int(duration * rate))
        return buffer.getvalue()

    return make
