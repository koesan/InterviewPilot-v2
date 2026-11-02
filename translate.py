"""Live translation (any interview language -> Turkish) of the interviewer's speech, beside the answers.

Runs on the asyncio loop in its own task with its own HTTP client, so it never delays an answer.
Final transcript pieces are collected and sent when the interviewer pauses (or a monologue grows
long); while one request streams, newer speech queues up and goes out as one piece, so the Turkish
never falls behind by more than one request. The model also cleans speech-to-text errors and joins
broken fragments, so the Turkish reads as clean sentences.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import anthropic
from google import genai
from google.genai import types


from config import (
    REQUEST_TIMEOUT_S,
    TRANSLATE_CONTEXT_CHARS,
    TRANSLATE_FLUSH_CHARS,
    TRANSLATE_MAX_TOKENS,
    Config,
    anthropic_headers,
)
from languages import english_name

log = logging.getLogger(__name__)

def _system(source: str, names: tuple[str, ...]) -> str:
    """Translation rules for speech in `source` (an English language name, e.g. "Spanish")."""
    hint = f"Silently fix them from the context and from these names: {', '.join(names)}. " if names else (
        "Silently fix them from the context. "
    )
    return (
        f"You translate live {source} speech from a job interview into Turkish, for a Turkish candidate who reads "
        f"it on screen while listening. The {source} comes from speech-to-text, so it has wrong words, missing "
        f"punctuation and sentences cut in two. {hint}"
        "Then write the meaning as natural, everyday Turkish "
        "sentences, the way a Turkish engineer would say it. Keep technical terms as Turkish IT people say them "
        "(API, mikroservis, backend, deadline). Don't summarize, don't add anything, don't answer questions in the "
        "text; translate them. Translate only what is inside <translate>. <earlier> is what was already "
        "translated, given only so you understand the topic: never translate or repeat any of it. "
        "Output only the Turkish text, with no notes, labels or quotation marks."
    )


class Translator:
    """Callbacks: on_start() opens a new Turkish paragraph, on_delta(text) streams into it."""

    def __init__(
        self,
        cfg: Config,
        on_start: Callable[[], None],
        on_delta: Callable[[str], None],
        on_notice: Callable[[str], None],
    ) -> None:
        self._client: anthropic.AsyncAnthropic | None = None
        self._models: list[str] = []
        if cfg.anthropic_key:
            self._client = anthropic.AsyncAnthropic(
                api_key=cfg.anthropic_key, max_retries=1, timeout=REQUEST_TIMEOUT_S,
                default_headers=anthropic_headers(cfg),
            )
            self._models = list(cfg.translate_models)
        # Gemini translates when Claude can't (bad key, no credit, outage), so the pane never stays empty.
        self._gemini: genai.Client | None = None
        self._gemini_model = cfg.gemini_model
        if cfg.gemini_key:
            self._gemini = genai.Client(
                api_key=cfg.gemini_key,
                http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_S * 1000)),
            )
            thinking = (  # 2.5: budget 0 = no thinking; 3.x Flash-Lite: minimal, or it thinks for seconds
                types.ThinkingConfig(thinking_budget=0) if cfg.gemini_model.startswith("gemini-2.5")
                else types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
            )
            self._gemini_thinking = thinking
        self._names = cfg.keyterms
        self.set_source("en")
        self._on_start = on_start
        self._on_delta = on_delta
        self._on_notice = on_notice
        self.enabled = True
        self._pending: list[str] = []     # final pieces not yet sent
        self._ready: list[str] = []       # pieces whose sentence is complete, waiting for the worker
        self._context = ""                # speech already translated, the tail is sent as context
        self._wake = asyncio.Event()
        self._generation = 0              # bumped on clear(), so a running request stops showing text
        self._warned = False

    def set_source(self, code: str) -> None:
        """The language the interviewer speaks (a key of LANGUAGES)."""
        self._system = _system(english_name(code), self._names)
        if self._gemini is not None:
            self._gemini_config = types.GenerateContentConfig(
                system_instruction=self._system,
                max_output_tokens=TRANSLATE_MAX_TOKENS,
                thinking_config=self._gemini_thinking,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

    # ---- input (asyncio thread) ---------------------------------------------------------

    def add(self, text: str) -> None:
        if not self.enabled or not (text := text.strip()):
            return
        self._pending.append(text)
        if sum(len(p) for p in self._pending) >= TRANSLATE_FLUSH_CHARS:
            self.flush()

    def flush(self) -> None:
        """The interviewer paused: send what has been said since the last piece."""
        if self._pending:
            self._ready.append(" ".join(self._pending))
            self._pending.clear()
            self._wake.set()

    def clear(self) -> None:
        """Drop everything (language switch or translation turned off)."""
        self._pending.clear()
        self._ready.clear()
        self._context = ""
        self._generation += 1

    # ---- worker -------------------------------------------------------------------------

    async def run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._ready:
                text = " ".join(self._ready)
                self._ready.clear()
                await self._translate(text)

    async def _translate(self, text: str) -> None:
        generation = self._generation
        context = self._context[-TRANSLATE_CONTEXT_CHARS:]
        user = (f"<earlier>{context}</earlier>\n\n" if context else "") + f"<translate>{text}</translate>"
        self._context = f"{self._context} {text}"[-TRANSLATE_CONTEXT_CHARS * 2:]
        started = False

        def emit(delta: str) -> bool:
            """Show a piece; False once the pane was cleared meanwhile."""
            nonlocal started
            if generation != self._generation:
                return False
            if not started:
                started = True
                self._on_start()
            self._on_delta(delta)
            return True

        for model in list(self._models):
            try:
                async with self._client.messages.stream(  # type: ignore[union-attr]
                    model=model,
                    max_tokens=TRANSLATE_MAX_TOKENS,
                    system=self._system,
                    messages=[{"role": "user", "content": user}],
                    thinking={"type": "disabled"},  # speed: a translation needs no reasoning
                ) as stream:
                    async for delta in stream.text_stream:
                        if not emit(delta):
                            return
                return
            except asyncio.CancelledError:
                raise
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
                self._claude_off(exc)
                break
            except anthropic.BadRequestError as exc:
                if "workspace" in str(exc).lower() or "credit" in str(exc).lower():
                    self._claude_off(exc)  # a key/account problem: every Claude model fails the same way
                    break
                log.warning("translate model %s unusable, dropping it: %r", model, exc)
                if len(self._models) > 1:
                    self._models.remove(model)
                if started:
                    return
            except anthropic.NotFoundError as exc:
                log.warning("translate model %s unusable, dropping it: %r", model, exc)
                if len(self._models) > 1:
                    self._models.remove(model)  # e.g. a model this key can't use: next one from now on
                if started:
                    return
            except Exception as exc:  # noqa: BLE001 - translation is a helper, never fatal
                log.warning("translate via %s failed: %r", model, exc)
                if started:
                    return
        if self._gemini is not None and not started:
            try:
                stream = await self._gemini.aio.models.generate_content_stream(
                    model=self._gemini_model, contents=user, config=self._gemini_config
                )
                async for chunk in stream:
                    if chunk.text and not emit(chunk.text):
                        return
                if started:
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("translate via Gemini failed: %r", exc)
            if started:
                return
        if not self._warned:
            self._warned = True
            self._on_notice("Çeviri alınamadı (cevaplar etkilenmez)")

    def _claude_off(self, exc: BaseException) -> None:
        """Same key for every Claude model: stop calling Claude, Gemini translates from now on."""
        log.warning("translate: Claude key rejected, using Gemini from now on: %r", exc)
        self._models.clear()
        self._on_notice("Claude anahtarı kullanılamıyor: çeviri Gemini ile yapılıyor")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._gemini is not None:
            try:
                await self._gemini.aio.aclose()
            except Exception:  # noqa: BLE001
                pass
