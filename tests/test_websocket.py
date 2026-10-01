import asyncio
import contextlib
import json
from uuid import UUID

import httpx
import pytest
from websockets.asyncio.client import connect as real_connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from qwen_reader.config import MODEL, MODEL_VOICES, Settings
from qwen_reader.synthesis import Synthesizer, wav_duration


@pytest.mark.parametrize("model", MODEL_VOICES)
async def test_native_websocket_protocol_and_model_voice_selection(tmp_path, monkeypatch, model):
    requests = []

    async def server(socket):
        assert socket.request.headers["Authorization"] == "Bearer test-key"
        run = json.loads(await socket.recv())
        requests.append(run)
        task = run["header"]["task_id"]
        UUID(task)
        assert run["header"]["action"] == "run-task"
        assert run["header"]["streaming"] == "duplex"
        payload = run["payload"]
        assert payload["model"] == model
        assert payload["input"] == {}
        assert payload["parameters"]["voice"] == MODEL_VOICES[model]
        assert payload["parameters"]["format"] == "pcm"
        assert payload["parameters"]["sample_rate"] == 24000
        await socket.send(json.dumps({"header": {"task_id": task, "event": "task-started"}}))
        continuation = json.loads(await socket.recv())
        assert continuation["header"]["action"] == "continue-task"
        assert continuation["header"]["task_id"] == task
        assert continuation["payload"]["input"]["text"] == "Hello"
        finish = json.loads(await socket.recv())
        assert finish["header"]["action"] == "finish-task"
        assert finish["header"]["task_id"] == task
        assert finish["payload"]["input"] == {}
        await socket.send(b"\0\0" * 24000)
        await socket.send(json.dumps({"header": {"task_id": task, "event": "task-finished"}}))

    settings = Settings(
        api_key="test-key",
        endpoint="wss://workspace.test/api-ws/v1/inference",
        model=model,
        cache_dir=tmp_path,
    )
    async with serve(server, "127.0.0.1", 0) as local_server:
        port = local_server.sockets[0].getsockname()[1]

        def connect(endpoint, **kwargs):
            assert endpoint == settings.endpoint
            return real_connect(f"ws://127.0.0.1:{port}", **kwargs)

        monkeypatch.setattr("qwen_reader.synthesis.connect", connect)
        async with httpx.AsyncClient() as client:
            synth = Synthesizer(settings, client)
            path = await synth.synthesize("Hello", lambda *args: None)
            assert wav_duration(path) == 1
            await synth.synthesize("Hello", lambda *args: None)
    assert len(requests) == 1


@pytest.mark.parametrize("failure", ["closed", "task-failed"])
async def test_websocket_failure_never_caches_incomplete_audio(tmp_path, monkeypatch, failure):
    async def server(socket):
        run = json.loads(await socket.recv())
        task = run["header"]["task_id"]
        await socket.send(json.dumps({"header": {"event": "task-started", "task_id": task}}))
        await socket.recv()
        await socket.recv()
        await socket.send(b"\0\0" * 24000)
        if failure == "task-failed":
            await socket.send(
                json.dumps(
                    {
                        "header": {
                            "event": "task-failed",
                            "task_id": task,
                            "error_code": "InvalidParameter",
                            "error_message": "Bad voice",
                        }
                    }
                )
            )

    async with serve(server, "127.0.0.1", 0) as local_server:
        port = local_server.sockets[0].getsockname()[1]
        monkeypatch.setattr(
            "qwen_reader.synthesis.connect",
            lambda _, **kw: real_connect(f"ws://127.0.0.1:{port}", **kw),
        )
        async with httpx.AsyncClient() as client:
            synth = Synthesizer(
                Settings(
                    api_key="key",
                    endpoint="wss://host.test/api-ws/v1/inference",
                    cache_dir=tmp_path,
                ),
                client,
            )
            with pytest.raises(ValueError, match="before completion|InvalidParameter"):
                await synth.synthesize("Hello", lambda *args: None)
    assert list(tmp_path.iterdir()) == []


async def test_closing_stream_closes_websocket_and_removes_partial_cache(tmp_path, monkeypatch):
    closed = asyncio.Event()

    async def server(socket):
        run = json.loads(await socket.recv())
        task = run["header"]["task_id"]
        await socket.send(json.dumps({"header": {"event": "task-started", "task_id": task}}))
        await socket.recv()
        await socket.recv()
        await socket.send(b"\0\0" * 24000)
        with contextlib.suppress(ConnectionClosed):
            await socket.recv()
        closed.set()

    async with serve(server, "127.0.0.1", 0) as local_server:
        port = local_server.sockets[0].getsockname()[1]
        monkeypatch.setattr(
            "qwen_reader.synthesis.connect",
            lambda _, **kw: real_connect(f"ws://127.0.0.1:{port}", **kw),
        )
        async with httpx.AsyncClient() as client:
            settings = Settings(
                api_key="key",
                endpoint="wss://host.test/api-ws/v1/inference",
                cache_dir=tmp_path / "cache",
            )
            synth = Synthesizer(settings, client)
            async with contextlib.aclosing(
                synth.stream("Hello", lambda *args: None, tmp_path / "spool")
            ) as stream:
                assert (await anext(stream)).duration == 0.75
            await asyncio.wait_for(closed.wait(), 2)
    assert list(settings.cache_dir.iterdir()) == []


def test_model_configuration_and_next_is_not_supported(monkeypatch):
    monkeypatch.delenv("QWEN_TTS_MODEL", raising=False)
    monkeypatch.delenv("QWEN_TTS_VOICE", raising=False)
    assert Settings.from_env().model == MODEL
    monkeypatch.setenv("QWEN_TTS_MODEL", "qwen-audio-3.0-tts-plus")
    assert Settings.from_env().voice_id == "longanlingxin"
    monkeypatch.setenv("QWEN_TTS_MODEL", "qwen-audio-3.1-tts-next")
    with pytest.raises(ValueError, match="Unknown model"):
        Settings.from_env().validate_tts()
