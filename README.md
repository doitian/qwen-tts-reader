# Qwen article reader

A Python / Textual TUI that extracts articles through [Defuddle](https://defuddle.md/docs#api-use), streams narration from Qwen Audio, and plays it through mpv. The default model is `qwen-audio-3.1-tts-flash`.

## Run

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and **mpv** on Linux or macOS. Install mpv with your system package manager (`brew install mpv`, `sudo apt install mpv`, or `sudo pacman -S mpv`).

```sh
uv sync
cp .env.example .env
# Edit .env: set QWEN_TTS_ENDPOINT and QWEN_TTS_API_KEY.
uv run qwen-reader
# Or start reading a URL directly:
uv run qwen-reader 'https://stephango.com/saw'
```

You can also install with `pip install -e .` and run `qwen-reader`. The app reads `.env`; existing environment variables take precedence. Never commit your key.

Configure both values explicitly in `.env`, replacing `YOUR_WORKSPACE` with your Beijing Model Studio workspace ID:

```dotenv
QWEN_TTS_ENDPOINT=wss://YOUR_WORKSPACE.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference
QWEN_TTS_API_KEY=your-api-key
QWEN_TTS_MODEL=qwen-audio-3.1-tts-flash
```

There is no default endpoint or legacy DashScope fallback. If you used the earlier POC configuration, replace `DASHSCOPE_API_KEY` with `QWEN_TTS_API_KEY` and set the full endpoint instead of `DASHSCOPE_WORKSPACE_ID`. Restart the reader after changing configuration.

Paste a URL, then press **Enter** or click **Read**. Playback starts after roughly 0.75 seconds of audio has arrived; the rest is generated in the background. The timeline shows received audio as **buffered** until the full duration is known. Press **Esc** to cancel generation and playback, or load another URL to replace the article. Completed requests remain cached for retries.

Starting `uv run qwen-reader` without a URL reopens the last article and restores its playback position, speed, and pause state. Progress is saved every two seconds, after playback controls, when switching articles, and on quit. Each article/narration keeps its own position. If audio must be regenerated, playback waits silently until the saved position is buffered. A finished article stays at the end, ready to replay with Space.

Bookmarks live in `playback.json` inside `QWEN_READER_CACHE_DIR`. They contain article URLs and playback settings, but no API keys or article text. Delete that file to clear saved progress. Changed article text, model, voice, or endpoint gets a separate bookmark because the audio timeline can differ.

| Control | Action |
| --- | --- |
| Space / Play button | Play, pause, or replay after the end |
| ← / → | Rewind / fast-forward 10 seconds |
| − / + (or =) | Adjust speed by 0.1×, from 0.5× to 3.0× |
| Click article text | Seek to the start of that paragraph |
| Scroll wheel / ↑ ↓ / Page Up, Page Down / Home, End / scrollbar | Browse freely; stop automatic following |
| Resume sync button | Bring the current paragraph into view and resume following |
| Ctrl+L | Focus the URL input |
| Model button / F2 | Choose a model and voice from dropdowns, or enter a custom voice ID |
| Esc | Cancel preparation / leave URL input |
| Q / Ctrl+C | Quit |

Playback shortcuts are inactive while editing the URL. Mouse buttons work too. Speed changes take effect immediately with pitch correction; they do not make another TTS request. Seeking preserves the paused state and works across received audio, including segment boundaries. Fast-forward clamps to the buffered end; playback waits there if more audio is still being generated. Once complete, the entire article is seekable.

The current **paragraph** is highlighted and kept in view. Paragraphs are the timing and seek units to avoid many tiny sentence requests; exceptionally long paragraphs are split near sentence boundaries. Highlighting follows mpv's audio position, so pause, speed changes, rewind, and replay stay aligned. Scrolling turns off automatic following while the highlight continues to show what is playing. It stays off until you click **Resume sync**, including when you click text to seek. Loading a new article starts with following enabled.

Clicking an unbuffered paragraph marks it in yellow and queues a seek; playback jumps there as soon as its first audio arrives. The status shows that it is waiting. Another text click replaces the pending selection; play/pause or a timed seek clears it. Clicking text preserves the current pause state.

## Models

Choose from the TUI's **Model** dialog, pass `--model`, or set `QWEN_TTS_MODEL` in `.env`:

| Model | Default voice |
| --- | --- |
| `qwen-audio-3.1-tts-flash` (default) | `longanlingxin_v3.1` |
| `qwen-audio-3.0-tts-plus` | `longanlingxin` |
| `qwen-audio-3.0-tts-flash` | `longanhuan_v3.6` |

```sh
uv run qwen-reader --model qwen-audio-3.0-tts-plus 'https://stephango.com/saw'
```

The **Voice** dropdown shows system voice names and IDs for the selected model. Choose **Default** to use its default voice, or **Custom voice ID…** for a cloned or additional base voice. Switching models resets the voice to a compatible default. The bundled system voice list comes from the [official voice catalog](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list).

Applying a different model or voice stops playback; press Read to use the selection. Reopening the last session without a URL also restores the model and voice used for that narration. Explicit `--model` / `--voice` options or changed environment configuration take precedence. Set environment variables for your default choices. These are the three supported streaming models; TTS Next is excluded because it does not stream.

## Configuration

| Environment variable | Default / purpose |
| --- | --- |
| `QWEN_TTS_ENDPOINT` | Required native WSS endpoint, including `/api-ws/v1/inference` |
| `QWEN_TTS_API_KEY` | Required API key for the configured endpoint |
| `QWEN_TTS_MODEL` | `qwen-audio-3.1-tts-flash`; one of the three models above |
| `QWEN_TTS_VOICE` | Optional override; leave blank for the selected model's default |
| `DEFUDDLE_API_KEY` | Optional key for additional Defuddle requests |
| `QWEN_READER_CACHE_DIR` | `$XDG_CACHE_HOME/qwen-tts-reader`, otherwise `~/.cache/qwen-tts-reader` |

`--voice VOICE` overrides `QWEN_TTS_VOICE`. `--model` selects that model's default voice unless `--voice` is also supplied. Voice IDs differ by model; see the [official voice list](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list).

The native [WebSocket protocol](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-client-events) sends `run-task`, waits for `task-started`, then sends `continue-task` and `finish-task`. Binary PCM frames arrive until `task-finished`. It requests 24 kHz, 16-bit mono PCM and writes its own valid WAV headers, avoiding the placeholder sizes in provider-generated WAV files that caused false truncation errors.

An explicitly configured HTTPS `/api/v1/services/audio/tts/SpeechSynthesizer` endpoint is also supported via the native [HTTP SSE API](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-http-api). It streams base64 PCM using the `X-DashScope-SSE: enable` protocol header. This is not an OpenAI or Anthropic compatibility API, and there is no fallback to the legacy DashScope host.

## POC behavior and limits

- The URL is sent to Defuddle; extracted prose is sent to Alibaba Cloud for synthesis. Standard service usage charges and rate limits apply.
- Markdown link labels and inline code are spoken; formatting, images, and fenced code blocks are skipped. No summarization is performed.
- Each paragraph is a separate streamed request, with a limit of 1,800 characters per request. One paragraph is prefetched ahead with a bounded queue (at most two requests in flight), and audio always plays in article order. Each request's initial 0.75-second segment is followed by two-second segments queued in mpv's playlist. Delivery may vary at paragraph boundaries. Network stalls or playback faster than generation can cause buffering.
- Completed text requests are cached and assembled into a full WAV for instant subsequent playback. An interrupted request is never cached as complete; already received audio remains playable with the error shown. Temporary playback segments are deleted when the article is replaced or the app exits.
- Audio is cached by text, model, voice, and endpoint. Actual audio durations and original text offsets are saved in a JSON seek index alongside the completed article WAV. Reopening an article still fetches its current text; unchanged content reuses its audio and timing. Audio cached before paragraph synchronization is regenerated once. Delete the cache directory to reclaim disk space. There is no automatic eviction.
- Static, publicly accessible articles work best. Login walls, paywalls, and some JavaScript pages may fail or yield incomplete extraction. There is no browser session or login integration.
- Linux and macOS only in this POC (mpv Unix IPC). A working local audio device is required. No word-level highlighting.

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests cover article parsing, paragraph timing, WebSocket and SSE streaming, bounded lookahead, cancellation, caches, highlighting, scrolling, seeking, restart restoration, model/voice selection, terminal resize, command-palette dismissal and quitting, and real mpv playback with null audio output. Tests never call paid TTS APIs. The mpv integration tests skip if mpv is absent.
