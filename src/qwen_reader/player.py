from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .synthesis import wav_duration

log = logging.getLogger(__name__)


class PlayerError(RuntimeError):
    pass


def find_mpv() -> str | None:
    if sys.platform == "win32":
        # mpv.com, which PATHEXT finds first, is a console wrapper; terminating it orphans mpv.exe.
        return shutil.which("mpv.exe")
    return shutil.which("mpv")


async def open_ipc_connection(address: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    if sys.platform != "win32":
        return await asyncio.open_unix_connection(address)
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(loop=loop)
    protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
    transport, _ = await loop.create_pipe_connection(lambda: protocol, address)
    return reader, asyncio.StreamWriter(transport, protocol, reader, loop)


class MpvPlayer:
    """A private mpv process controlled over JSON IPC: a Unix socket, or a Windows named pipe."""

    def __init__(self, *, audio_output: str | None = None, log_file: Path | None = None):
        self.audio_output = audio_output
        self.log_file = log_file
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
        executable = find_mpv()
        if not executable:
            raise PlayerError(
                "mpv is required. Install it with brew install mpv, scoop install mpv, "
                "or your package manager."
            )
        if sys.platform == "win32":
            address = rf"\\.\pipe\qwen-mpv-{uuid.uuid4().hex}"
        else:
            # Short base directory: Unix socket paths are limited to about 104 bytes.
            self.directory = tempfile.TemporaryDirectory(prefix="qwen-mpv-", dir="/tmp")
            address = str(Path(self.directory.name) / "ipc.sock")
        args = [
            executable,
            "--no-config",
            "--idle=yes",
            "--no-video",
            "--no-terminal",
            "--keep-open=yes",
            "--gapless-audio=yes",
            "--audio-pitch-correction=yes",
            "--input-default-bindings=no",
            f"--input-ipc-server={address}",
        ]
        if self.audio_output:
            args.append(f"--ao={self.audio_output}")
        if self.log_file:
            args.append(f"--log-file={self.log_file}")
        log.info("starting %s with IPC %s", executable, address)
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
                        self.reader, self.writer = await open_ipc_connection(address)
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
            started = time.monotonic()
            try:
                async with asyncio.timeout(3):
                    self.writer.write(
                        (json.dumps({"command": command, "request_id": request_id}) + "\n").encode()
                    )
                    await self.writer.drain()
                    while line := await self.reader.readline():
                        response = json.loads(line)
                        if "event" in response:
                            log.debug("mpv event %s", line.decode().strip())
                        if response.get("request_id") != request_id:
                            continue
                        elapsed = time.monotonic() - started
                        if elapsed > 0.1:
                            log.warning("slow mpv command %s took %.3fs", command, elapsed)
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

    async def wait_loaded(self, path: Path, index: int | None = None) -> bool:
        """Wait until `path` is the current file with audio ready.

        Returns False if mpv already moved past playlist entry `index`: a file shorter
        than its audio buffer is decoded at once, even paused, and gapless playback then
        makes the next file current while the short one's audio is still queued.
        """
        # loadfile is asynchronous; wait until the file and audio output are ready.
        try:
            async with asyncio.timeout(8):
                while True:
                    try:
                        loaded_path = await self.command("get_property", "path")
                        if loaded_path == str(path.resolve()):
                            await self.command("get_property", "time-pos")
                            if await self.command("get_property", "audio-out-params"):
                                return True
                        elif index is not None and (
                            await self.command("get_property", "playlist-pos") > index
                        ):
                            return False
                    except PlayerError as exc:
                        if "property unavailable" not in str(exc):
                            raise
                    if await self.command("get_property", "idle-active"):
                        raise PlayerError("mpv could not open the audio. Check your sound device.")
                    await asyncio.sleep(0.05)
        except TimeoutError as exc:
            raise PlayerError("mpv did not load the audio in time. Restart the reader.") from exc

    async def append(self, path: Path, duration: float) -> None:
        """Queue the next segment; resume automatically if playback ran out of audio."""
        async with self.playback_lock:
            try:
                ended = await self.command("get_property", "eof-reached")
            except PlayerError as exc:
                if "property unavailable" not in str(exc):
                    raise
                # Briefly unavailable while mpv switches files, when audio hasn't run out.
                ended = False
            index = await self.command("get_property", "playlist-pos")
            was_last = index == len(self.parts) - 1
            await self.command("loadfile", str(path.resolve()), "append")
            self.parts.append((path.resolve(), duration))
            log.debug(
                "appended %s (%.2fs) as part %d while playing part %d",
                path.name,
                duration,
                len(self.parts) - 1,
                index,
            )
            if ended and was_last:
                log.warning("playback had run out of audio; restarting at part %d", index + 1)
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
            # mpv has no position for a few milliseconds while it switches files.
            for attempt in range(20):
                if attempt:
                    await asyncio.sleep(0.01)
                index = await self.command("get_property", "playlist-pos")
                try:
                    ended = await self.command("get_property", "eof-reached")
                    position = await self.part_position(ended)
                except PlayerError as exc:
                    if "property unavailable" not in str(exc):
                        raise
                    continue
                if index == await self.command("get_property", "playlist-pos"):
                    offset = sum(duration for _, duration in self.parts[: max(0, index)])
                    return (
                        max(0.0, offset + float(position or 0)),
                        self.user_paused,
                        bool(ended and index == len(self.parts) - 1),
                    )
            raise PlayerError("mpv could not read the playback position.")

    async def part_position(self, ended: bool) -> float | None:
        # After a gapless switch, time-pos reads 0 in the next part while the previous part's
        # last ~0.25s is still audible; audio-pts stays negative until that audio has played.
        # audio-pts is unavailable at EOF, and briefly during the switch itself.
        return await self.command("get_property", "time-pos" if ended else "audio-pts")

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
            current = True
            if index != await self.command("get_property", "playlist-pos"):
                await self.command("set_property", "pause", True)
                await self.command("playlist-play-index", index)
                current = await self.wait_loaded(self.parts[index][0], index)
            # A part mpv already moved past is too short to seek within; play it from its start.
            if current:
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
