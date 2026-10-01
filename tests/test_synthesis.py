import asyncio
import json
import sys

import httpx
import pytest

from qwen_reader.__main__ import main
from qwen_reader.config import MODEL, Settings
from qwen_reader.synthesis import Synthesizer, join_wavs, purge_cache, valid_wav, wav_duration


async def test_api_contract_audio_concatenation_and_cache(tmp_path, sse_audio, tts_endpoint):
    posts = []

    def handler(request):
        if request.method == "POST":
            assert str(request.url) == tts_endpoint
            assert request.url.path == "/api/v1/services/audio/tts/SpeechSynthesizer"
            assert request.headers["Authorization"] == "Bearer secret"
            assert request.headers["X-DashScope-SSE"] == "enable"
            payload = json.loads(request.content)
            assert payload["model"] == MODEL == "qwen-audio-3.1-tts-flash"
            assert payload["input"]["voice"] == "longanlingxin_v3.1"
            assert payload["input"]["format"] == "pcm"
            assert payload["input"]["sample_rate"] == 24000
            assert payload["input"]["rate"] == 1.0
            posts.append(payload["input"]["text"])
            return sse_audio()
        pytest.fail("Streaming synthesis must not wait for an audio URL download")

    settings = Settings(api_key="secret", endpoint=tts_endpoint, cache_dir=tmp_path, chunk_chars=20)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        synth = Synthesizer(settings, client)
        progress = []
        text = "Sentence one. Sentence two. Sentence three."
        path = await synth.synthesize(text, lambda *args: progress.append(args))
        assert len(posts) == 3
        assert wav_duration(path) == 3
        assert await synth.synthesize(text, lambda *args: None) == path
        assert len(posts) == 3
        # Rebuild a missing joined file using cached chunks, without another API call.
        path.unlink()
        await synth.synthesize(text, lambda *args: None)
        assert len(posts) == 3
        assert progress[-1][:2] == (3, 3)


@pytest.mark.parametrize(
    "status,payload,message",
    [
        (401, {"code": "InvalidApiKey", "message": "Bad key"}, "InvalidApiKey"),
        (200, {"code": "ModelNotFound", "message": "No such model"}, "ModelNotFound"),
        (200, {"output": {}}, "did not return SSE"),
        (200, ["unexpected"], "unexpected response"),
    ],
)
async def test_api_errors(tmp_path, status, payload, message, tts_endpoint):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=payload))
    ) as client:
        with pytest.raises(ValueError, match=message):
            settings = Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path)
            await Synthesizer(settings, client).synthesize("Hello", lambda *args: None)


async def test_cancellation_does_not_cache_partial_audio(tmp_path, tts_endpoint):
    started = asyncio.Event()

    async def handler(request):
        started.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        settings = Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path)
        synth = Synthesizer(settings, client)
        task = asyncio.create_task(synth.synthesize("Hello", lambda *args: None))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert list(tmp_path.iterdir()) == []


async def test_retry_on_rate_limit(tmp_path, sse_audio, tts_endpoint):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"code": "Throttling"})
        return sse_audio()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        settings = Settings(api_key="test", endpoint=tts_endpoint, cache_dir=tmp_path)
        path = await Synthesizer(settings, client).synthesize("Hello", lambda *args: None)
    assert attempts == 2
    assert valid_wav(path)


def test_truncated_wav_is_not_a_cache_hit(tmp_path, wav_bytes):
    path = tmp_path / "broken.wav"
    path.write_bytes(wav_bytes()[:-10])
    assert not valid_wav(path)


def test_mismatched_formats_do_not_publish_an_article(tmp_path, wav_bytes):
    paths = [tmp_path / "one.wav", tmp_path / "two.wav"]
    paths[0].write_bytes(wav_bytes(rate=24000))
    paths[1].write_bytes(wav_bytes(rate=16000))
    destination = tmp_path / "result.wav"
    with pytest.raises(ValueError, match="different audio formats"):
        join_wavs(paths, destination)
    assert not destination.exists()
    assert sorted(tmp_path.iterdir()) == paths


def test_configured_endpoint_and_key_are_used_and_secrets_are_hidden(monkeypatch, tts_endpoint):
    monkeypatch.setenv("QWEN_TTS_ENDPOINT", f" {tts_endpoint} ")
    monkeypatch.setenv("QWEN_TTS_API_KEY", " very-secret ")
    settings = Settings.from_env()
    settings.validate_tts()
    assert settings.endpoint == tts_endpoint
    assert settings.api_key == "very-secret"
    assert "very-secret" not in repr(settings)


def test_legacy_environment_does_not_supply_an_endpoint_or_key(monkeypatch):
    monkeypatch.delenv("QWEN_TTS_ENDPOINT", raising=False)
    monkeypatch.delenv("QWEN_TTS_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "old-workspace")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "old-key")
    settings = Settings.from_env()
    assert settings.endpoint == ""
    assert settings.api_key == ""
    with pytest.raises(ValueError, match="QWEN_TTS_ENDPOINT and QWEN_TTS_API_KEY"):
        settings.validate_tts()


def test_missing_key_is_actionable(tts_endpoint):
    with pytest.raises(ValueError, match="QWEN_TTS_API_KEY"):
        Settings(endpoint=tts_endpoint).validate_tts()


async def test_missing_endpoint_is_reported_before_network_request(tmp_path):
    def handler(request):
        pytest.fail("No request should be sent without an explicit endpoint")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="QWEN_TTS_ENDPOINT"):
            await Synthesizer(Settings(api_key="test", cache_dir=tmp_path), client).synthesize(
                "Hello", lambda *args: None
            )


def test_purge_cache_deletes_only_reader_files_and_keeps_bookmarks(tmp_path, monkeypatch, capsys):
    speech = ["chunk-a.wav", "article-b.wav", "article-b.json", "tmpx.partial", "tmpy.wav"]
    for name in [*speech, "playback.json", ".playback-z.tmp", "notes.txt", "chunk-c.txt"]:
        (tmp_path / name).write_bytes(b"x" * 1000)
    monkeypatch.setenv("QWEN_READER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["qwen-reader", "purge-cache"])
    main()
    assert f"Removed 5 files (0.0 MB) from {tmp_path}." in capsys.readouterr().out
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ".playback-z.tmp",
        "chunk-c.txt",
        "notes.txt",
        "playback.json",
    ]
    assert purge_cache(tmp_path, state=True) == (2, 2000, [])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["chunk-c.txt", "notes.txt"]
