# InterviewPilot

**English** · [Türkçe](README.tr.md)

A real-time desktop assistant for online job interviews. It listens to the call audio, transcribes the interviewer live, and when you press **Space** it shows a short, natural answer based on your CV and the job description.

![Demo](docs/demo.gif)

▶ [Full demo video (MP4, with sound)](docs/demo.mp4): an English interview clip is played, the question *"Tell me something about yourself"* is transcribed and translated, and an answer appears in under a second.

## Features

- **Live transcription:** captures system audio (Teams, Zoom, Meet, browser, any app) with Deepgram Nova-3 streaming. Your microphone is never recorded.
- **Instant answers:** Claude streams the reply; the answer is usually prepared while the interviewer is still finishing, so it can appear the moment you press Space. Automatic fallback to a faster Claude model and Gemini.
- **10 interview languages:** English, Turkish, Mandarin Chinese, Hindi, Spanish, French, Arabic, Bengali, Portuguese and Russian. Speech recognition and answers switch together.
- **Live Turkish translation** of the interviewer's speech in a separate pane, without slowing the answers.
- **Interview memory:** the whole conversation is sent as a dialogue, so the model keeps context and does not repeat examples. Prompts are cached to cut cost.
- **Correction bar:** type a note (e.g. a misheard term or "shorter") and get a new answer to the same question.
- **Thinking modes:** off (fastest) or low / medium / high reasoning for harder questions.
- **Session logs:** each session writes a timestamped transcript and an answers file.

| Listening | Answer |
|---|---|
| ![Listening](docs/screenshot-listening.png) | ![Answer](docs/screenshot-answer.png) |

## Requirements

- Linux with PulseAudio or PipeWire (`parec` from `pulseaudio-utils` / `pipewire-pulse`)
- Python 3.11+
- API keys: [Deepgram](https://deepgram.com) (required) and [Anthropic](https://console.anthropic.com) and/or [Google Gemini](https://aistudio.google.com)

## Installation

```bash
git clone https://github.com/koesan/InterviewPilot-v2.git
cd InterviewPilot-v2
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then add your API keys
./run.sh
```

## Configuration

The model reads three texts from `promptlar/`. Copy the examples and edit them with your own information:

```bash
cd promptlar
cp cv.example.txt cv.txt          # your CV
cp job.example.txt job.txt        # the company and the role
cp prompt.example.txt prompt.txt  # answer style and rules (optional)
```

The `*_tr.txt` files are used in Turkish mode; every other language uses the English files. Your own `*.txt` files are git-ignored; when one is missing, the `.example.txt` version is used.

Optional settings in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_MODEL` | `claude-sonnet-5` | Main answer model |
| `CLAUDE_BACKUP_MODEL` | `claude-haiku-4-5-20251001` | Fast backup (`none` to disable) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Last fallback |
| `TRANSLATE_MODEL` | `claude-haiku-5-5` | Translation model |
| `DEEPGRAM_MODEL` | `nova-3` | Speech-to-text model |
| `AUDIO_SOURCE` | `@DEFAULT_MONITOR@` | Audio source (`pactl list short sources`) |
| `DEEPGRAM_KEYTERMS` | | Comma-separated names and technical terms to boost recognition |

## Usage

1. Start the app before the call; it opens in **LISTENING** mode.
2. Select the interview language from the top-left menu.
3. When the interviewer finishes a question, press **Space**. The answer appears on the right and listening pauses.
4. Press **Space** again to continue listening.
5. If an answer misses the point, press **Ctrl+E**, type a note and press **Enter**.

Space works only while the app window is focused. For a global hotkey, bind `toggle.sh` in your window manager, e.g. in Hyprland:

```
bind = , F8, exec, /path/to/InterviewPilot-v2/toggle.sh
```

## Project structure

| File | Role |
|---|---|
| `main.py` | Entry point: Qt on the main thread, audio and network on an asyncio thread |
| `core.py` | Orchestration: listening state, early answers, retries |
| `audio.py` | System audio capture with `parec` |
| `stt.py` | Deepgram streaming client with reconnects |
| `llm.py` | Answer generation, prompt caching and provider fallback |
| `translate.py` | Live translation pane |
| `conversation.py` | Interview history |
| `languages.py` | Supported languages |
| `ui.py`, `style.qss` | Interface and dark theme |
| `records.py` | Session transcript and answer logs |
| `config.py` | Settings and tunables |

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests
```

## Disclaimer

This project is intended for interview practice and research. Check the rules of any interview you take part in and use it responsibly.

## License

[MIT](LICENSE)
