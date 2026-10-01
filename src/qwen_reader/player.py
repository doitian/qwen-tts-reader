from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .synthesis import wav_duration


class PlayerError(RuntimeError):
    pass


class MpvPlayer:
    """A private mpv process controlled over its JSON IPC Unix socket."""

    def __init__(self, *, audio_output: str | None = None):
        self.audio_output = audio_output
        self.process: asyncio.subprocess.Process | None = None
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.directory: tempfile.TemporaryDirectory | None = None
        self.lock = asyncio.Lock()
        self.playback_lock = asyncio.Lock()
        self.request_id = 0
        self.parts: list[tuple[Path, float]] = []
        self.user_paused = False

    async def start(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return
        if not shutil.which("mpv"):
            raise PlayerError(
                "mpv is required. Install it with brew install mpv or your package manager."
            )
        self.directory = tempfile.TemporaryDirectory(prefix="qwen-mpv-", dir="/tmp")
        socket = Path(self.directory.name) / "ipc.sock"
        args = [
            "mpv",
            "--no-config",
            "--idle=yes",
            "--no-video",
            "--no-terminal",
            "--keep-open=yes",
            "--gapless-audio=yes",
            "--audio-pitch-correction=yes",
            "--input-default-bindings=no",
            f"--input-ipc-server={socket}",
        ]
        if self.audio_output:
            args.append(f"--ao={self.audio_output}")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            async with asyncio.timeout(5):
                while True:
                    if self.process.returncode is not None:
                        raise PlayerError("mpv exited during startup. Check your mpv installation.")
                    try:
                        self.reader, self.writer = await asyncio.open_unix_connection(str(socket))
                        break
                    except (FileNotFoundError, ConnectionRefusedError):
                        await asyncio.sleep(0.05)
        except BaseException:
            await self.close()
            raise

    async def command(self, *command: Any) -> Any:
        async with self.lock:
            if self.writer is None or self.reader is None:
                raise PlayerError("The audio player is not running.")
            self.request_id += 1
            request_id = self.request_id
            try:
                async with asyncio.timeout(3):
                    self.writer.write(
                        (json.dumps({"command": command, "request_id": request_id}) + "\n").encode()
                    )
                    await self.writer.drain()
                    while line := await self.reader.readline():
                        response = json.loads(line)
                        if response.get("request_id") != request_id:
                            continue
                        if response.get("error") != "success":
                            raise PlayerError(f"mpv: {response.get('error', 'unknown error')}")
                        return response.get("data")
                    raise PlayerError("mpv disconnected. Restart the reader.")
            except (TimeoutError, ConnectionError, OSError) as exc:
                raise PlayerError("mpv is not responding. Restart the reader.") from exc

    async def load(self, path: Path, speed: float, *, paused: bool = False) -> None:
        async with self.playback_lock:
            await self.command("stop")
            self.parts = [(path.resolve(), wav_duration(path))]
            self.user_paused = paused
            await self.command("set_property", "pause", True)
            await self.command("set_property", "speed", speed)
            await self.command("loadfile", str(path.resolve()), "replace")
            await self.wait_loaded(path)
            await self.command("set_property", "pause", paused)

    async def wait_loaded(self, path: Path) -> None:
        # loadfile is asynchronous; wait until the file and audio output are ready.
        async with asyncio.timeout(8):
            while True:
                try:
                    loaded_path = await self.command("get_property", "path")
                    if loaded_path == str(path.resolve()):
                        await self.command("get_property", "time-pos")
                        if await self.command("get_property", "audio-out-params"):
                            return
                except PlayerError as exc:
                    if "property unavailable" not in str(exc):
                        raise
                if await self.command("get_property", "idle-active"):
                    raise PlayerError("mpv could not open the audio. Check your sound device.")
                await asyncio.sleep(0.05)

    async def append(self, path: Path, duration: float) -> None:
        """Queue the next segment; resume automatically if playback ran out of audio."""
        async with self.playback_lock:
            ended = await self.command("get_property", "eof-reached")
            index = await self.command("get_property", "playlist-pos")
            was_last = index == len(self.parts) - 1
            await self.command("loadfile", str(path.resolve()), "append")
            self.parts.append((path.resolve(), duration))
            if ended and was_last:
                await self.command("playlist-play-index", index + 1)
                await self.wait_loaded(path)
                await self.command("set_property", "pause", self.user_paused)

    async def stop(self) -> None:
        async with self.playback_lock:
            if self.writer:
                await self.command("stop")
            self.parts.clear()
            self.user_paused = False

    async def status(self) -> tuple[float, bool, bool]:
        async with self.playback_lock:
            # A native playlist transition may occur between IPC replies; retry its snapshot.
            for _ in range(3):
                index = await self.command("get_property", "playlist-pos")
                try:
                    position = await self.command("get_property", "time-pos")
                    ended = await self.command("get_property", "eof-reached")
                except PlayerError as exc:
                    if "property unavailable" not in str(exc):
                        raise
                    continue
                if index == await self.command("get_property", "playlist-pos"):
                    offset = sum(duration for _, duration in self.parts[: max(0, index)])
                    return (
                        offset + float(position or 0),
                        self.user_paused,
                        bool(ended and index == len(self.parts) - 1),
                    )
            raise PlayerError("mpv could not read the playback position.")

    async def set_paused(self, paused: bool) -> None:
        async with self.playback_lock:
            self.user_paused = paused
            await self.command("set_property", "pause", paused)

    async def seek(self, position: float) -> None:
        async with self.playback_lock:
            total = sum(duration for _, duration in self.parts)
            remaining = max(0, min(position, total - 0.001))
            index = 0
            for index, (_, duration) in enumerate(self.parts):
                if remaining < duration or index == len(self.parts) - 1:
                    break
                remaining -= duration
            if index != await self.command("get_property", "playlist-pos"):
                await self.command("set_property", "pause", True)
                await self.command("playlist-play-index", index)
                await self.wait_loaded(self.parts[index][0])
            await self.command("seek", remaining, "absolute+exact")
            await self.command("set_property", "pause", self.user_paused)

    async def set_speed(self, speed: float) -> None:
        await self.command("set_property", "speed", speed)

    async def close(self) -> None:
        if self.writer:
            self.writer.close()
            with contextlib.suppress(OSError):
                await self.writer.wait_closed()
            self.writer = None
            self.reader = None
        if self.process and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()
        self.process = None
        if self.directory:
            self.directory.cleanup()
            self.directory = None
