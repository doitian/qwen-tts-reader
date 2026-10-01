from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import tempfile
import uuid
import wave
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from .config import Settings
from .narration import Cue, narration_spans

SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # signed 16-bit mono PCM


@dataclass(frozen=True)
class AudioPart:
    path: Path
    duration: float


def write_pcm_wav(path: Path, pcm: bytes) -> None:
    if not pcm or len(pcm) % 2:
        raise ValueError("TTS returned incomplete PCM audio frames.")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Parse SSE records, including multiline data and a final unterminated record."""
    lines = []
    async for line in response.aiter_lines():
        if not line:
            if lines:
                yield "\n".join(lines)
                lines.clear()
        elif line.startswith("data:"):
            lines.append(line[5:].removeprefix(" "))
    if lines:
        yield "\n".join(lines)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if not audio.getnframes() or audio.getcomptype() != "NONE":
            raise ValueError("TTS returned an empty or unsupported WAV file.")
        expected = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
        size = 0
        while data := audio.readframes(65536):
            size += len(data)
        if size != expected:
            raise ValueError("TTS returned a truncated WAV file.")
        return audio.getnframes() / audio.getframerate()


def valid_wav(path: Path) -> bool:
    try:
        wav_duration(path)
        return True
    except (OSError, ValueError, wave.Error, EOFError):
        return False


def join_wavs(paths: list[Path], destination: Path) -> None:
    """Concatenate PCM frames, writing one correct WAV header atomically."""
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".wav", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with wave.open(str(temporary), "wb") as output:
            params = None
            for path in paths:
                with wave.open(str(path), "rb") as source:
                    current = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                    if params is None:
                        params = current
                        output.setnchannels(current[0])
                        output.setsampwidth(current[1])
                        output.setframerate(current[2])
                    elif params != current:
                        raise ValueError(
                            "TTS chunks have different audio formats. Retry the article."
                        )
                    while data := source.readframes(65536):
                        output.writeframesraw(data)
        wav_duration(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class Synthesizer:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    def cache_path(self, text: str, kind: str) -> Path:
        identity = json.dumps(
            [
                3,
                self.settings.model,
                self.settings.endpoint,
                self.settings.voice_id,
                SAMPLE_RATE,
                "pcm",
                text,
            ],
            ensure_ascii=False,
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return self.settings.cache_dir / f"{kind}-{digest}.wav"

    async def synthesize(self, text: str, progress: Callable[[int, int, str], None]) -> Path:
        """Convenience API for callers that want the completed, cached article."""
        self.settings.validate_tts()
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.settings.cache_dir) as directory:
            async for _ in self.stream(text, progress, Path(directory)):
                pass
        return self.cache_path(text, "article")

    def cached_cues(self, path: Path, text: str) -> list[Cue] | None:
        """A full narration cache is usable only with its matching seek index."""
        if not valid_wav(path):
            return None
        try:
            data = json.loads(path.with_suffix(".json").read_text())
            cues = [Cue(**entry) for entry in data["cues"]]
            duration = wav_duration(path)
            if data["version"] != 1 or not cues:
                return None
            previous_end = 0
            previous_time = 0.0
            for cue in cues:
                if not (
                    previous_end <= cue.start < cue.end <= len(text)
                    and not text[previous_end : cue.start].strip()
                    and abs(cue.time - previous_time) < 0.001
                    and cue.end_time is not None
                    and cue.time < cue.end_time <= duration + 0.001
                ):
                    return None
                previous_end, previous_time = cue.end, cue.end_time
            if text[previous_end:].strip() or abs(previous_time - duration) > 0.001:
                return None
            return cues
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def save_cues(self, path: Path, cues: list[Cue]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, suffix=".partial", delete=False
        ) as temp:
            temporary = Path(temp.name)
            json.dump({"version": 1, "cues": [asdict(cue) for cue in cues]}, temp)
        try:
            temporary.replace(path.with_suffix(".json"))
        finally:
            temporary.unlink(missing_ok=True)

    async def stream(
        self,
        text: str,
        progress: Callable[[int, int, str], None],
        spool_dir: Path,
        on_cue: Callable[[Cue], None] | None = None,
    ) -> AsyncIterator[AudioPart]:
        """Stream paragraphs in order, with one paragraph of bounded lookahead.

        Qwen may merge sentence events, so each request is one reading unit.
        Actual PCM durations provide exact boundaries without estimating speech
        speed or depending on voice-specific word timestamps.
        """
        self.settings.validate_tts()
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        spool_dir.mkdir(parents=True, exist_ok=True)
        spans = narration_spans(text, self.settings.chunk_chars)
        if not spans:
            raise ValueError("There is no article text to read.")
        destination = self.cache_path(text, "article")
        cached = self.cached_cues(destination, text)
        if cached is not None:
            progress(len(spans), len(spans), "Using cached narration")
            if on_cue:
                for cue in cached:
                    on_cue(cue)
            yield AudioPart(destination, wav_duration(destination))
            return

        queues: dict[int, asyncio.Queue] = {}
        tasks: dict[int, asyncio.Task] = {}
        locks: dict[Path, asyncio.Lock] = {}

        async def produce(index: int, queue: asyncio.Queue) -> None:
            span = spans[index]
            content = text[span.start : span.end]
            stream = self.stream_paragraph(content, spool_dir, index)
            lock = locks.setdefault(self.cache_path(content, "chunk"), asyncio.Lock())
            try:
                async with lock, contextlib.aclosing(stream):
                    async for part in stream:
                        await queue.put(part)
            except Exception as exc:
                await queue.put(exc)
            else:
                await queue.put(None)

        def start(index: int) -> None:
            if index < len(spans) and index not in tasks:
                # Two segments of lookahead per request, never an entire article.
                queues[index] = asyncio.Queue(maxsize=2)
                tasks[index] = asyncio.create_task(produce(index, queues[index]))

        paths = []
        cues = []
        elapsed = 0.0
        start(0)
        start(1)
        try:
            for index, span in enumerate(spans):
                progress(
                    index, len(spans), f"Receiving speech · paragraph {index + 1} of {len(spans)}"
                )
                cue = Cue(span.start, span.end, elapsed)
                if on_cue:
                    on_cue(cue)
                while True:
                    part = await queues[index].get()
                    if part is None:
                        break
                    if isinstance(part, Exception):
                        raise part
                    elapsed += part.duration
                    yield part
                cue = Cue(span.start, span.end, cue.time, elapsed)
                cues.append(cue)
                if on_cue:
                    on_cue(cue)
                paths.append(self.cache_path(text[span.start : span.end], "chunk"))
                await tasks.pop(index)
                del queues[index]
                start(index + 2)
                progress(index + 1, len(spans), f"Prepared {index + 1} of {len(spans)} paragraphs")
            await asyncio.to_thread(join_wavs, paths, destination)
            self.save_cues(destination, cues)
        finally:
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def stream_paragraph(
        self,
        text: str,
        spool_dir: Path,
        index: int,
    ) -> AsyncIterator[AudioPart]:
        path = self.cache_path(text, "chunk")
        if valid_wav(path):
            yield AudioPart(path, wav_duration(path))
            return
        with tempfile.NamedTemporaryFile(
            dir=self.settings.cache_dir, suffix=".partial", delete=False
        ) as temp:
            temporary = Path(temp.name)
        try:
            pending = bytearray()
            total = 0
            part_number = 0
            with wave.open(str(temporary), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(SAMPLE_RATE)
                pcm_stream = self.stream_pcm(text)
                async with contextlib.aclosing(pcm_stream):
                    async for pcm in pcm_stream:
                        total += len(pcm)
                        if total > 100 * 1024 * 1024:
                            raise ValueError("TTS audio chunk exceeds the 100 MB limit.")
                        output.writeframesraw(pcm)
                        pending.extend(pcm)
                        target = int(BYTES_PER_SECOND * (0.75 if part_number == 0 else 2))
                        while len(pending) >= target:
                            segment = spool_dir / f"paragraph-{index:06}-{part_number:06}.wav"
                            write_pcm_wav(segment, bytes(pending[:target]))
                            del pending[:target]
                            part_number += 1
                            yield AudioPart(segment, target / BYTES_PER_SECOND)
                            target = BYTES_PER_SECOND * 2
                if not total or total % 2:
                    raise ValueError("TTS returned empty or incomplete PCM audio frames.")
            wav_duration(temporary)
            temporary.replace(path)
            if pending:
                segment = spool_dir / f"paragraph-{index:06}-{part_number:06}.wav"
                write_pcm_wav(segment, bytes(pending))
                yield AudioPart(segment, len(pending) / BYTES_PER_SECOND)
        finally:
            temporary.unlink(missing_ok=True)

    def check_payload(self, payload: object, status: int = 200) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Qwen returned an unexpected response format.")
        if not 200 <= status < 300 or payload.get("code"):
            code = str(payload.get("code") or status)
            message = str(payload.get("message") or "Check your key, region, and voice.")
            detail = f"{code} — {message[:300]}".replace(self.settings.api_key, "[redacted]")
            raise ValueError(f"Qwen TTS: {detail}")
        return payload

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        if self.settings.endpoint.startswith("wss://"):
            stream = self.websocket_pcm(text)
        else:
            stream = self.sse_pcm(text)
        async with contextlib.aclosing(stream):
            async for pcm in stream:
                yield pcm

    async def websocket_pcm(self, text: str) -> AsyncIterator[bytes]:
        """run-task → task-started → continue-task/finish-task → PCM → task-finished."""
        task_id = uuid.uuid4().hex

        def message(action: str, payload: dict) -> str:
            return json.dumps(
                {
                    "header": {"action": action, "task_id": task_id, "streaming": "duplex"},
                    "payload": payload,
                }
            )

        try:
            async with connect(
                self.settings.endpoint,
                additional_headers={"Authorization": f"Bearer {self.settings.api_key}"},
                open_timeout=15,
                close_timeout=2,
                max_size=8 * 1024 * 1024,
            ) as websocket:
                await websocket.send(
                    message(
                        "run-task",
                        {
                            "task_group": "audio",
                            "task": "tts",
                            "function": "SpeechSynthesizer",
                            "model": self.settings.model,
                            "parameters": {
                                "text_type": "PlainText",
                                "voice": self.settings.voice_id,
                                "format": "pcm",
                                "sample_rate": SAMPLE_RATE,
                                "rate": 1.0,
                            },
                            "input": {},
                        },
                    )
                )
                started = False
                received_audio = False
                while True:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=60)
                    if isinstance(raw, bytes):
                        if not started:
                            raise ValueError("Qwen sent audio before starting the TTS task.")
                        received_audio = received_audio or bool(raw)
                        yield raw
                        continue
                    try:
                        event = json.loads(raw)
                        header = event["header"]
                        if header.get("task_id") != task_id:
                            raise ValueError("Qwen returned an unrelated task ID.")
                        kind = header["event"]
                    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
                        raise ValueError("Qwen returned a malformed WebSocket event.") from exc
                    if kind == "task-failed":
                        code = str(header.get("error_code", "TaskFailed"))
                        detail = str(header.get("error_message", "Speech generation failed."))
                        error = f"Qwen TTS: {code} — {detail[:300]}"
                        raise ValueError(error.replace(self.settings.api_key, "[redacted]"))
                    if kind == "task-started" and not started:
                        started = True
                        await websocket.send(message("continue-task", {"input": {"text": text}}))
                        await websocket.send(message("finish-task", {"input": {}}))
                    elif kind == "task-finished":
                        if not started or not received_audio:
                            raise ValueError("Qwen returned no audio in its stream.")
                        return
        except ConnectionClosed as exc:
            raise ValueError("TTS stream ended before completion. Retry the article.") from exc
        except (WebSocketException, OSError, TimeoutError) as exc:
            raise ValueError(
                "TTS WebSocket connection failed or timed out. Check endpoint and key."
            ) from exc

    async def sse_pcm(self, text: str) -> AsyncIterator[bytes]:
        """Native Qwen SSE: base64 PCM, with explicit successful completion required."""
        for attempt in range(3):
            async with self.client.stream(
                "POST",
                self.settings.endpoint,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "X-DashScope-SSE": "enable",
                },
                json={
                    "model": self.settings.model,
                    "input": {
                        "text": text,
                        "voice": self.settings.voice_id,
                        "format": "pcm",
                        "sample_rate": SAMPLE_RATE,
                        "rate": 1.0,
                    },
                },
                timeout=httpx.Timeout(60, connect=15),
                follow_redirects=False,
            ) as response:
                if response.status_code in (429, 503) and attempt < 2:
                    try:
                        delay = float(response.headers.get("retry-after", 2 ** (attempt + 1)))
                    except ValueError:
                        delay = 2 ** (attempt + 1)
                else:
                    if not response.is_success or "text/event-stream" not in response.headers.get(
                        "content-type", ""
                    ):
                        await response.aread()
                        try:
                            self.check_payload(response.json(), response.status_code)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Qwen returned an invalid response (HTTP {response.status_code})."
                            ) from exc
                        raise ValueError("The endpoint did not return SSE audio. Check its URL.")
                    finished = False
                    received_audio = False
                    async for event in sse_data(response):
                        if event == "[DONE]":
                            break
                        try:
                            payload = self.check_payload(json.loads(event))
                            output = payload.get("output") or {}
                            data = (output.get("audio") or {}).get("data")
                            if data:
                                pcm = base64.b64decode(data, validate=True)
                                received_audio = received_audio or bool(pcm)
                                yield pcm
                            if output.get("finish_reason") == "stop":
                                finished = True
                                break
                        except (
                            json.JSONDecodeError,
                            binascii.Error,
                            AttributeError,
                            TypeError,
                        ) as exc:
                            raise ValueError("Qwen returned a malformed audio stream.") from exc
                    if not finished:
                        raise ValueError("TTS stream ended before completion. Retry the article.")
                    if not received_audio:
                        raise ValueError("Qwen returned no audio in its stream.")
                    return
            # Retry only rejected requests, never an interrupted stream that already played.
            await asyncio.sleep(max(0, min(delay, 30)))
