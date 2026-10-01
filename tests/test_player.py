import asyncio
import time

import pytest

from qwen_reader.player import MpvPlayer, find_mpv


@pytest.mark.skipif(not find_mpv(), reason="mpv is not installed")
async def test_load_paused_restores_without_playing_from_the_beginning(tmp_path, wav_bytes):
    path = tmp_path / "resume.wav"
    path.write_bytes(wav_bytes(5))
    player = MpvPlayer(audio_output="null")
    try:
        await player.start()
        await player.load(path, 1.4, paused=True)
        await asyncio.sleep(0.15)
        position, paused, _ = await player.status()
        assert paused
        assert position == pytest.approx(0, abs=0.05)
        assert await player.command("get_property", "pause") is True
        await player.seek(3.25)
        position, paused, _ = await player.status()
        assert position == pytest.approx(3.25, abs=0.2)
        assert paused
        await player.set_paused(False)
        await asyncio.sleep(0.15)
        assert (await player.status())[0] > 3.25
    finally:
        await player.close()


@pytest.mark.skipif(not find_mpv(), reason="mpv is not installed")
async def test_real_mpv_play_pause_seek_speed_and_cleanup(tmp_path, wav_bytes):
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes(20))
    player = MpvPlayer(audio_output="null")
    try:
        await player.start()
        process = player.process
        socket_dir = player.directory.name if player.directory else None
        await player.load(path, 1.0)
        await asyncio.sleep(0.15)
        position, paused, ended = await player.status()
        assert position >= 0
        assert not paused
        assert not ended
        await player.set_paused(True)
        await player.seek(10)
        await asyncio.sleep(0.1)
        position, paused, _ = await player.status()
        assert position == pytest.approx(10, abs=0.2)
        assert paused
        await player.set_speed(1.8)
        assert await player.command("get_property", "speed") == 1.8
        assert await player.command("get_property", "audio-pitch-correction") is True
        await player.seek(2)
        await asyncio.sleep(0.1)
        # mpv's pitch-correction buffer can shift the reported position by a few tenths.
        assert (await player.status())[0] == pytest.approx(2, abs=0.5)
        await player.set_paused(False)
        await player.seek(19.9)
        await asyncio.sleep(0.5)
        assert (await player.status())[2] is True
        await player.seek(0)
        await player.set_paused(False)
        await asyncio.sleep(0.1)
        assert (await player.status())[2] is False
        await player.stop()
        await player.load(path, 1.3)
        assert (await player.status())[1] is False
    finally:
        await player.close()
    from pathlib import Path

    assert process.returncode is not None
    assert socket_dir is None or not Path(socket_dir).exists()


@pytest.mark.skipif(not find_mpv(), reason="mpv is not installed")
async def test_streaming_queue_global_seek_pause_and_resume_from_live_edge(tmp_path, wav_bytes):
    paths = [tmp_path / f"part-{i}.wav" for i in range(3)]
    for path in paths:
        path.write_bytes(wav_bytes(1))
    player = MpvPlayer(audio_output="null")
    try:
        await player.start()
        await player.load(paths[0], 1)
        await player.append(paths[1], 1)
        await player.set_paused(True)
        await player.seek(1.5)
        await asyncio.sleep(0.1)
        position, paused, ended = await player.status()
        assert position == pytest.approx(1.5, abs=0.2)
        assert paused and not ended
        assert await player.command("get_property", "pause") is True
        await player.seek(0.4)
        await asyncio.sleep(0.1)
        assert (await player.status())[0] == pytest.approx(0.4, abs=0.2)
        await player.set_paused(False)
        await player.seek(1.9)
        await asyncio.sleep(0.5)
        assert (await player.status())[2] is True
        # mpv pauses itself at EOF; this is buffering, not a user-requested pause.
        assert (await player.status())[1] is False
        await player.set_paused(True)
        await player.append(paths[2], 1)
        assert (await player.status())[1] is True
        assert await player.command("get_property", "pause") is True
        await player.set_speed(1.5)
        await player.set_paused(False)
        await asyncio.sleep(0.15)
        assert (await player.status())[0] >= 2
        await player.seek(0)
        assert (await player.status())[2] is False
    finally:
        await player.close()


@pytest.mark.skipif(not find_mpv(), reason="mpv is not installed")
async def test_position_is_continuous_across_gapless_part_transitions(tmp_path, wav_bytes):
    paths = [tmp_path / f"part-{i}.wav" for i in range(6)]
    for path in paths:
        path.write_bytes(wav_bytes(0.5))
    player = MpvPlayer(audio_output="null")
    try:
        await player.start()
        await player.load(paths[0], 1)
        started = time.monotonic()
        for path in paths[1:]:
            await player.append(path, 0.5)
        drift = []
        previous = 0.0
        # Polling every few milliseconds lands inside mpv's brief file-switch windows.
        while (elapsed := time.monotonic() - started) < 2.2:
            position, _, ended = await player.status()
            assert position >= previous - 0.02
            previous = position
            drift.append(position - elapsed)
            await asyncio.sleep(0.003)
        assert not ended
        assert max(drift) - min(drift) < 0.15
    finally:
        await player.close()
