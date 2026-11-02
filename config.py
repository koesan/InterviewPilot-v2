"""Settings: API keys from .env, CV text and latency tunables."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
PROMPTS_DIR = BASE_DIR / "promptlar"  # every text the model reads: rules, job, CV
CV_FILE = PROMPTS_DIR / "cv.txt"
JOB_FILE = PROMPTS_DIR / "job.txt"
PROMPT_FILE = PROMPTS_DIR / "prompt.txt"
CV_TR_FILE = PROMPTS_DIR / "cv_tr.txt"
JOB_TR_FILE = PROMPTS_DIR / "job_tr.txt"
PROMPT_TR_FILE = PROMPTS_DIR / "prompt_tr.txt"
STYLE_FILE = BASE_DIR / "style.qss"
LOG_FILE = BASE_DIR / "assistant.log"
RECORDS_DIR = BASE_DIR / "kayitlar"   # one transcript file and one answers file per session
PID_FILE = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "interview-assistant.pid"

# Audio
SAMPLE_RATE = 16_000
AUDIO_READ_BYTES = 1_280            # up to 40 ms of s16le mono per read
AUDIO_LATENCY_MS = 20

# Deepgram
DG_ENDPOINTING_MS = 300
DG_UTTERANCE_END_MS = 1_000
DG_KEEPALIVE_AFTER_S = 4.0          # Deepgram drops idle streams after ~10 s
FINALIZE_WAIT_S = 0.4               # max wait for flushed transcript, then use the interim text
ABSORB_LATE_FINAL_S = 3.0           # drop flush results that arrive after we used the interim

# Conversation memory: the whole interview is sent as a dialogue (cached, ~10% price on repeats)
SPEECH_MAX_CHARS = 6_000            # interviewer speech since the last answer (a ~4 min monologue)
HISTORY_MAX_CHARS = 60_000          # earlier exchanges (~15k tokens, about an hour); oldest dropped beyond it
MIN_QUESTION_WORDS = 3

# LLM
MAX_TOKENS_CLAUDE = 900               # room for live-coding answers; a cap only, short replies stay short
MAX_TOKENS_GEMINI = 2_048           # includes Gemini's internal thinking tokens
HEDGE_AFTER_S = 2.0                 # primary still silent (a real stall) -> start the fast backup
FIRST_TOKEN_TIMEOUT_S = 5.0
REQUEST_TIMEOUT_S = 20.0
CLAUDE_COOLDOWN_S = 60.0            # after rate limit / overload
CLAUDE_DISABLE_S = 1_800.0          # after auth / billing errors
WARM_IDLE_S = 30.0                  # re-warm HTTP connection when idle longer
CACHE_REFRESH_S = 240.0             # prompt cache lives 5 min; refresh it before it expires

# A screenshot sent with the question: the image adds ~1.5-2k input tokens, so the first word comes a little
# later; a screen task (code, a matrix, a long question) also needs more room to answer.
SHOT_MAX_TOKENS_CLAUDE = 1_600
SHOT_EXTRA_WAIT_S = 2.0             # added to the hedge and first-token timeouts when an image is attached
SHOT_PREVIEW_MS = 10_000            # the thumbnail stays this long once the window is in front

# Thinking modes for the primary model, picked in the UI: label -> (effort, max_tokens, first-token
# timeout, hedge). Thinking happens before the first visible word, so the timeouts grow with effort;
# otherwise the no-thinking backup would win the race and the thinking would be wasted.
THINKING_OFF = "Kapalı (hızlı)"
THINKING_LEVELS: dict[str, tuple[str | None, int, float, float]] = {
    THINKING_OFF: (None, MAX_TOKENS_CLAUDE, FIRST_TOKEN_TIMEOUT_S, HEDGE_AFTER_S),
    "Low": ("low", 3_000, 15.0, 10.0),
    "Medium": ("medium", 6_000, 30.0, 22.0),
    "High": ("high", 12_000, 50.0, 40.0),
}

# Answer models picked in the UI: label -> model id ("gemini" = GEMINI_MODEL). Only the picked model gets the
# request; when it fails (bad key, no credit, overload, no first word in time) the next one in its chain answers.
GEMINI = "gemini"
MODEL_CHOICES: dict[str, str] = {
    "Sonnet 5": "claude-sonnet-5",
    "Sonnet 5.5": "claude-sonnet-5-5",
    "Opus 5.5": "claude-opus-5-5",
    "Haiku 5.5": "claude-haiku-5-5",
    "Haiku 4.5": "claude-haiku-4-5-20251001",
    "Gemini": GEMINI,
}
# How each model is told not to think (the API differs per model; the default is {"type": "disabled"}).
THINKING_OFF_PARAMS: dict[str, dict] = {
    "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}},  # "disabled" is a 400 on Sonnet 5.5
    "claude-opus-5-5": {"output_config": {"effort": "low"}},       # Opus 5.5 always thinks; low is the least
}
NO_ADAPTIVE_THINKING = frozenset({"claude-haiku-4-5-20251001"})    # Haiku 4.5 answers without thinking

# Live translation of the transcript into Turkish of the transcript (separate, cheap model; never blocks answers)
TRANSLATE_MAX_TOKENS = 800
TRANSLATE_CONTEXT_CHARS = 600       # earlier English sent as context only, so names and topic stay right
TRANSLATE_FLUSH_CHARS = 280         # translate a long monologue in pieces instead of waiting for a pause

log = logging.getLogger(__name__)


def anthropic_headers(cfg: "Config") -> dict[str, str] | None:
    """An organization-wide key must name its workspace on every request."""
    return {"anthropic-workspace-id": cfg.anthropic_workspace} if cfg.anthropic_workspace else None


@dataclass(frozen=True)
class Config:
    deepgram_key: str
    anthropic_key: str
    anthropic_workspace: str
    gemini_key: str
    claude_model: str
    claude_backup_model: str
    translate_models: tuple[str, ...]
    gemini_model: str
    audio_source: str
    deepgram_model: str
    keyterms: tuple[str, ...]
    cv_text: str
    job_text: str = ""
    prompt_text: str = ""
    cv_text_tr: str = ""
    job_text_tr: str = ""
    prompt_text_tr: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _read_text(path: Path) -> str:
    """Your own file (git-ignored) wins; the committed *.example.txt is the fallback."""
    if not path.exists():
        path = path.with_name(f"{path.stem}.example{path.suffix}")
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        log.error("%s okunamadı: %s", path.name, exc)
        return ""


def load_config() -> Config:
    load_dotenv(ENV_FILE)
    cfg = Config(
        deepgram_key=_env("DEEPGRAM_API_KEY"),
        anthropic_key=_env("ANTHROPIC_API_KEY"),
        anthropic_workspace=_env("ANTHROPIC_WORKSPACE_ID"),  # only for keys not scoped to a workspace
        gemini_key=_env("GEMINI_API_KEY"),
        claude_model=_env("CLAUDE_MODEL", "claude-sonnet-5"),
        claude_backup_model=_env("CLAUDE_BACKUP_MODEL", "claude-haiku-4-5-20251001").replace("none", ""),
        translate_models=tuple(dict.fromkeys(
            m for m in (_env("TRANSLATE_MODEL", "claude-haiku-5-5"), "claude-haiku-4-5-20251001") if m
        )),
        gemini_model=_env("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        audio_source=_env("AUDIO_SOURCE", "@DEFAULT_MONITOR@"),
        deepgram_model=_env("DEEPGRAM_MODEL", "nova-3"),
        keyterms=tuple(t.strip() for t in _env("DEEPGRAM_KEYTERMS").split(",") if t.strip()),
        cv_text=_read_text(CV_FILE),
        job_text=_read_text(JOB_FILE),
        prompt_text=_read_text(PROMPT_FILE),
        cv_text_tr=_read_text(CV_TR_FILE),
        job_text_tr=_read_text(JOB_TR_FILE),
        prompt_text_tr=_read_text(PROMPT_TR_FILE),
    )
    warnings = []
    if not cfg.deepgram_key:
        warnings.append("DEEPGRAM_API_KEY eksik (.env)")
    if not cfg.anthropic_key and not cfg.gemini_key:
        warnings.append("ANTHROPIC_API_KEY ve GEMINI_API_KEY eksik (.env)")
    if not cfg.prompt_text:
        warnings.append("prompt.txt boş veya yok — kısa yedek kurallar kullanılıyor")
    if not cfg.cv_text:
        warnings.append("cv.txt boş veya yok — cevaplar CV'ye dayanmayacak")
    return Config(**{**cfg.__dict__, "warnings": tuple(warnings)})
