from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import logging
import tempfile
import time
import uuid
import wave
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from .config import Settings
from .narration import Cue, TextSpan, narration_spans

SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # signed 16-bit mono PCM

log = logging.getLogger(__name__)


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


# Only names the reader creates, since the cache directory may be shared.
SPEECH_FILES = ("chunk-*.wav", "article-*.wav", "article-*.json", "tmp*.partial", "tmp*.wav")
STATE_FILES = ("playback.json", ".playback-*.tmp")


def purge_cache(cache_dir: Path, *, state: bool = False) -> tuple[int, int, list[Path]]:
    """Delete cached speech, and bookmarks too if `state`.

    Returns how many files were removed, the bytes freed, and files left because
    they are in use, such as audio a running reader is playing.
    """
    removed = freed = 0
    busy = []
    for pattern in SPEECH_FILES + (STATE_FILES if state else ()):
        for path in cache_dir.glob(pattern):
            if not path.is_file():
                continue
            size = path.stat().st_size
            try:
                path.unlink()
            except PermissionError:
                busy.append(path)
                continue
            removed += 1
            freed += size
    return removed, freed, busy


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

    def known_durations(self, text: str) -> dict[int, float]:
        """Durations of already cached reading units, read from WAV headers only."""
        known = {}
        for index, span in enumerate(narration_spans(text, self.settings.chunk_chars)):
            path = self.cache_path(text[span.start : span.end], "chunk")
            with contextlib.suppress(OSError, EOFError, wave.Error), wave.open(str(path)) as audio:
                if audio.getnframes():
                    known[index] = audio.getnframes() / audio.getframerate()
        return known

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

    @staticmethod
    def join_chunks(spans: list[TextSpan], paths: list[Path], destination: Path) -> list[Cue]:
        """Publish the full article once every paragraph is cached; [] if any is missing."""
        if not all(valid_wav(path) for path in paths):
            return []
        cues = []
        elapsed = 0.0
        for span, path in zip(spans, paths, strict=True):
            duration = wav_duration(path)
            cues.append(Cue(span.start, span.end, elapsed, elapsed + duration))
            elapsed += duration
        join_wavs(paths, destination)
        return cues

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
        *,
        first: int = 0,
        room: Callable[[], bool] | None = None,
    ) -> AsyncIterator[AudioPart]:
        """Stream paragraphs in order from `first`, with one paragraph of bounded lookahead.

        Earlier paragraphs are never requested. A new request starts only while
        `room()` allows more buffered audio; a started request always finishes.
        Cue times count from `first`, or from 0 when the whole article is cached.

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

        def has_room(index: int) -> bool:
            # Cached units cost no request, so they never wait for room.
            if room is None or index >= len(spans):
                return True
            cached = self.cache_path(text[spans[index].start : spans[index].end], "chunk")
            return cached.exists() or room()

        first = max(0, min(first, len(spans) - 1))
        elapsed = 0.0
        start(first)
        if has_room(first + 1):
            start(first + 1)
        try:
            for index in range(first, len(spans)):
                span = spans[index]
                if index not in tasks:
                    # Room follows the continuously moving playback position; poll it.
                    while not has_room(index):  # noqa: ASYNC110
                        await asyncio.sleep(0.25)
                    start(index)
                progress(
                    index, len(spans), f"Receiving speech · paragraph {index + 1} of {len(spans)}"
                )
                cue = Cue(span.start, span.end, elapsed)
                if on_cue:
                    on_cue(cue)
                while True:
                    waiting = time.monotonic()
                    part = await queues[index].get()
                    if (waited := time.monotonic() - waiting) > 0.05:
                        log.info("waited %.2fs for audio from paragraph %d", waited, index)
                    if part is None:
                        break
                    if isinstance(part, Exception):
                        raise part
                    elapsed += part.duration
                    yield part
                cue = Cue(span.start, span.end, cue.time, elapsed)
                if on_cue:
                    on_cue(cue)
                await tasks.pop(index)
                del queues[index]
                # Fill the lookahead in order, so a later paragraph never jumps the queue.
                for ahead in (index + 1, index + 2):
                    if ahead not in tasks and not has_room(ahead):
                        break
                    start(ahead)
                progress(index + 1, len(spans), f"Prepared {index + 1} of {len(spans)} paragraphs")
            # Paragraphs may have been cached across sessions and in any order.
            paths = [self.cache_path(text[span.start : span.end], "chunk") for span in spans]
            if cues := await asyncio.to_thread(self.join_chunks, spans, paths, destination):
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
            log.debug("paragraph %d: cached chunk %s", index, path.name)
            yield AudioPart(path, wav_duration(path))
            return
        requested = time.monotonic()
        log.debug("paragraph %d: requesting %d characters", index, len(text))
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
                        if not total:
                            log.debug(
                                "paragraph %d: first audio after %.2fs",
                                index,
                                time.monotonic() - requested,
                            )
                        total += len(pcm)
                        if total > 100 * 1024 * 1024:
                            raise ValueError("TTS audio chunk exceeds the 100 MB limit.")
                        output.writeframesraw(pcm)
                        pending.extend(pcm)
                        target = int(BYTES_PER_SECOND * (0.75 if part_number == 0 else 2))
                        # Later cuts keep half a second back, so a paragraph never ends in a
                        # sliver that mpv skips past; the first cut keeps playback starting fast.
                        tail = 0 if part_number == 0 else BYTES_PER_SECOND // 2
                        while len(pending) >= target + tail:
                            segment = spool_dir / f"paragraph-{index:06}-{part_number:06}.wav"
                            write_pcm_wav(segment, bytes(pending[:target]))
                            del pending[:target]
                            part_number += 1
                            yield AudioPart(segment, target / BYTES_PER_SECOND)
                            target, tail = BYTES_PER_SECOND * 2, BYTES_PER_SECOND // 2
                if not total or total % 2:
                    raise ValueError("TTS returned empty or incomplete PCM audio frames.")
            wav_duration(temporary)
            temporary.replace(path)
            elapsed = time.monotonic() - requested
            log.info(
                "paragraph %d: %.2fs of audio in %.2fs (%.1fx real time)",
                index,
                total / BYTES_PER_SECOND,
                elapsed,
                total / BYTES_PER_SECOND / elapsed,
            )
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
