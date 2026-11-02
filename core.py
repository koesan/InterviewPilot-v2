"""Orchestrator: runs on the asyncio thread, talks to the UI only through Qt signals.

Latency strategy: while LISTENING, as soon as the interviewer pauses (Deepgram
UtteranceEnd, or an endpoint on a sentence ending in "?") the answer is generated
speculatively in the background. Pressing Space then shows it instantly if no new
speech arrived; otherwise the stale job is cancelled and a fresh one starts.

While PAUSED, retry() re-answers that same question with a note typed by hand - a garbled
term, or a direction such as "give an example" - for when the answer missed the point.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from array import array
from dataclasses import dataclass, field
from typing import Any, Coroutine

from audio import AudioCapture
from config import (
    MODEL_CHOICES,
    FINALIZE_WAIT_S,
    HISTORY_MAX_CHARS,
    MIN_QUESTION_WORDS,
    RECORDS_DIR,
    SPEECH_MAX_CHARS,
    Config,
)
from conversation import Candidate, Conversation
from languages import LANGUAGES, TRANSLATION_TARGET, display_name
from llm import AnswerEngine, AnswerError
from records import SessionRecord
from screenshot import CaptureError, Shot, capture
from stt import DeepgramSTT
from translate import Translator

log = logging.getLogger(__name__)

SOUND_PEAK = 300  # s16 peak above ~-40 dBFS counts as "audio playing"
CARD_NOTE_CHARS = 48  # the note shown on the answer card, so the question stays readable
REJECTED_KEPT = 3     # earlier wrong replies to the same question, shown to the model on a correction
REJECTED_CHARS = 240  # their opening is enough to show which reading was wrong


@dataclass
class _Job:
    candidate: Candidate             # carries the interview history as it was when the job was built
    note: str = ""                   # free text typed on the correction bar (retries only)
    retry: bool = False              # replaces the previous answer instead of following it
    image: Shot | None = None        # their shared screen, sent with the speech
    rejected: tuple[str, ...] = ()   # replies to this question the candidate corrected (retries only)
    task: asyncio.Task[None] | None = None
    chunks: list[str] = field(default_factory=list)
    first_token_at: float | None = None
    provider: str | None = None
    error: str | None = None
    answer_id: int | None = None     # set once the answer is shown in the UI
    shown_at: float = 0.0


class Core:
    """`ui` is the Qt Bridge; emitting its signals from this thread is queued safely."""

    def __init__(self, cfg: Config, ui: Any) -> None:
        self._cfg = cfg
        self._ui = ui
        self._listening = True
        self._answer_seq = 0
        self._spec: _Job | None = None
        self._last: _Job | None = None      # the answer on screen, reusable for a correction
        self._retry: _Job | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._last_sound = 0.0
        self._sound_active = False
        self._pause_lock = asyncio.Lock()  # rapid Space presses must not interleave
        # Screenshot: the first left-Alt tap arms, the user switches to the shared screen, the second captures.
        # The picture waits here and goes with the next Space answer.
        self._shot: Shot | None = None
        self._shot_armed = False
        self._shot_task: asyncio.Task[None] | None = None
        self._conv = Conversation(SPEECH_MAX_CHARS, HISTORY_MAX_CHARS, MIN_QUESTION_WORDS)
        self._engine = AnswerEngine(cfg, on_provider=ui.provider.emit)
        self._record = SessionRecord(RECORDS_DIR)
        log.info("records: %s, %s", self._record.transcript_path.name, self._record.answers_path.name)
        self._stt: DeepgramSTT | None = None
        if cfg.deepgram_key:
            self._stt = DeepgramSTT(
                cfg.deepgram_key,
                cfg.keyterms,
                on_interim=ui.interim.emit,
                on_final=self._on_final,
                on_utterance_end=self._on_utterance_end,
                on_status=ui.stt_status.emit,
                model=cfg.deepgram_model,
            )
        self._audio = AudioCapture(cfg.audio_source, self._on_audio, ui.notice.emit)
        self._lang = "en"
        self._translate_wanted = True       # the translation pane is open; nothing to translate in Turkish mode
        self._translator: Translator | None = None
        if cfg.anthropic_key or cfg.gemini_key:
            self._translator = Translator(
                cfg, ui.translation_started.emit, ui.translation_delta.emit, ui.notice.emit
            )

    def set_thinking(self, label: str) -> None:
        self._engine.set_thinking(label)
        if self._engine.primary_ready():
            self._ui.notice.emit(f"Düşünme: {label}")
        else:  # Gemini, Haiku 4.5, or a Claude model paused after an error: they answer without thinking
            self._ui.notice.emit(f"Düşünme: {label} — seçili model şu an düşünmeyi kullanamıyor (etkisiz)")

    def set_model(self, label: str) -> None:
        """The model picked in the UI answers alone; the chain behind it only takes over on an error."""
        model = MODEL_CHOICES.get(label)
        if model is None:
            return
        self._engine.set_model(model)
        if self._engine.selected_model() != model:
            self._ui.notice.emit(f"{label} için API anahtarı yok (.env)")
            return
        self._drop_spec()  # an answer being prepared by the old model is dropped
        self._engine.warm(self._conv.history())  # write the prompt cache for the new model now
        self._ui.notice.emit(f"Model: {label}")

    def set_translation(self, wanted: bool) -> None:
        """The translation pane was shown or hidden; a hidden pane costs nothing."""
        self._translate_wanted = wanted
        self._sync_translator()

    def _sync_translator(self) -> None:
        if self._translator is None:
            return
        enabled = self._translate_wanted and self._lang != TRANSLATION_TARGET
        if self._translator.enabled and not enabled:
            self._translator.clear()
        self._translator.enabled = enabled

    async def set_language(self, code: str) -> None:
        """Switch the interview language (a key of LANGUAGES): reconnects STT, switches prompts, resets history."""
        if code not in LANGUAGES or code == self._lang:
            return
        self._lang = code
        log.info("language -> %s", code)

        # Switch LLM prompts
        self._engine.set_language(code)

        # Reconnect STT with new language
        if self._stt is not None:
            await self._stt.reconnect_with_language(code)

        # Speech in the target language is never translated; any other language is
        if self._translator is not None:
            self._translator.clear()
            self._translator.set_source(code)
        self._sync_translator()

        # Reset conversation — mixing languages would confuse the model
        self._conv = Conversation(SPEECH_MAX_CHARS, HISTORY_MAX_CHARS, MIN_QUESTION_WORDS)
        self._spec = None
        self._last = None
        self._retry = None
        self._engine.warm()  # the other language's prompt is not cached yet: write it now, not on the first question

        self._ui.notice.emit(f"Dil: {display_name(code)}")

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            log.error("background task failed", exc_info=exc)
            self._ui.notice.emit(f"Beklenmeyen hata: {exc}")

    # ---- lifecycle ------------------------------------------------------------------

    async def run(self) -> None:
        self._ui.listening_changed.emit(self._listening)
        for warning in self._cfg.warnings:
            self._ui.notice.emit(warning)
        self._spawn(self._startup_check())
        if self._translator is not None:
            self._spawn(self._translator.run())
        if self._stt is None:
            self._ui.stt_status.emit("Deepgram: anahtar yok", False)
            return
        self._spawn(self._stt.run())
        self._spawn(self._audio.run())

    async def _startup_check(self) -> None:
        for problem in await self._engine.check():
            self._ui.notice.emit(problem)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._audio.close()
        if self._stt is not None:
            await self._stt.close()
        await self._engine.close()
        if self._translator is not None:
            await self._translator.close()
        self._record.close()

    # ---- audio / transcript -----------------------------------------------------------

    async def _on_audio(self, chunk: bytes) -> None:
        self._track_level(chunk)
        # PAUSED: audio is read (keeps parec's buffer fresh) but never leaves the machine.
        if self._listening and self._stt is not None:
            await self._stt.send_audio(chunk)

    def _track_level(self, chunk: bytes) -> None:
        """Tell the UI whether any system audio is playing (for "is Teams audio captured?")."""
        samples = array("h", chunk[: len(chunk) & ~1])
        now = time.monotonic()
        if samples and max(max(samples), -min(samples)) > SOUND_PEAK:
            self._last_sound = now
        active = now - self._last_sound < 1.5
        if active != self._sound_active:
            self._sound_active = active
            self._ui.audio_active.emit(active)

    def _on_final(self, text: str, end_of_turn: bool) -> None:
        self._conv.add(text)
        self._ui.final.emit(text, end_of_turn)
        if not text:
            return
        self._record.speech(text)
        if self._translator is not None:
            self._translator.add(text)
            if end_of_turn:
                self._translator.flush()
        self._engine.warm(self._conv.history())
        self._drop_stale_speculation()
        if self._listening and end_of_turn and text.endswith("?"):
            self._speculate()

    def _on_utterance_end(self) -> None:
        if self._translator is not None:
            self._translator.flush()
        if self._listening:
            self._speculate()

    # ---- speculative answers ------------------------------------------------------------

    def _drop_spec(self) -> None:
        if self._spec is not None and self._spec.answer_id is None:
            self._spec.task.cancel()  # type: ignore[union-attr]
            self._spec = None

    def _drop_stale_speculation(self) -> None:
        job = self._spec
        if job is not None and job.answer_id is None:
            current = self._conv.peek(allow_empty=self._shot is not None)
            if current is None or current.key != job.candidate.key or job.image is not self._shot:
                job.task.cancel()  # type: ignore[union-attr]
                self._spec = None

    def _speculate(self) -> None:
        candidate = self._conv.peek(allow_empty=self._shot is not None)
        if candidate is None:
            return
        spec = self._spec
        if spec is not None and spec.candidate.key == candidate.key and spec.image is self._shot:
            return  # already generating for exactly this speech and screen
        if spec is not None and spec.answer_id is None:
            spec.task.cancel()  # type: ignore[union-attr]
        log.info("speculating on %r%s", candidate.question[-80:], " + screen" if self._shot else "")
        self._spec = self._start_job(candidate, image=self._shot)

    # ---- screenshot (left Alt via Hyprland + shot.sh / button) ---------------------------------------------

    def screenshot(self) -> None:
        """First press arms, second press captures the monitor in front of the user.

        Called from the Qt thread via call_soon_threadsafe, like toggle().
        """
        if self._shot_task is not None and not self._shot_task.done():
            return  # still capturing
        if not self._shot_armed:
            self._shot_armed = True
            self._set_shot(None)  # a new capture replaces the old (probably wrong) one
            self._ui.screenshot_state.emit("armed")
            self._ui.notice.emit("Ekran: paylaşılan ekrana geç, sol Alt'a tekrar bas")
            return
        self._shot_armed = False
        self._ui.screenshot_state.emit("busy")
        self._shot_task = self._spawn(self._capture())

    async def _capture(self) -> None:
        try:
            shot = await capture()
        except CaptureError as exc:
            log.error("screenshot failed: %s", exc)
            self._ui.screenshot_state.emit("idle")
            self._ui.notice.emit(f"Ekran görüntüsü alınamadı: {exc}")
            return
        # Every capture is kept in the session folder, also the wrong ones that are never sent.
        shot = dataclasses.replace(shot, file=self._record.screenshot(shot.data, shot.media_type))
        self._set_shot(shot)
        self._ui.screenshot_taken.emit(shot.data)
        self._ui.screenshot_state.emit("ready")
        self._ui.notice.emit("Ekran alındı · Space ile konuşmayla birlikte gider")

    def clear_screenshot(self) -> None:
        """Right click on the button: drop the waiting picture or cancel the armed capture."""
        self._shot_armed = False
        self._set_shot(None)
        self._ui.screenshot_state.emit("cleared")
        self._ui.notice.emit("Ekran görüntüsü iptal edildi")

    def _set_shot(self, shot: Shot | None) -> None:
        self._shot = shot
        self._drop_stale_speculation()  # an answer prepared without (or with another) screen is stale
        if shot is not None and self._listening:
            self._speculate()  # start reading the screen now, so Space is fast

    # ---- Space toggle -------------------------------------------------------------------

    def toggle(self) -> None:
        self._listening = not self._listening
        log.info("state -> %s", "LISTENING" if self._listening else "PAUSED")
        self._ui.listening_changed.emit(self._listening)
        if not self._listening:
            self._spawn(self._on_pause(time.monotonic()))

    async def _on_pause(self, t_pause: float) -> None:
        async with self._pause_lock:
            await self._handle_pause(t_pause)

    async def _handle_pause(self, t_pause: float) -> None:
        if self._stt is not None:
            interim = await self._stt.finalize(FINALIZE_WAIT_S)
            if interim:  # flush was slow: use what is already on screen
                self._conv.add(interim)
                self._ui.final.emit(interim, True)
                self._record.speech(interim)
                if self._translator is not None:
                    self._translator.add(interim)
        if self._translator is not None:
            self._translator.flush()
        self._ui.paragraph_break.emit()

        image = self._shot
        candidate = self._conv.peek(allow_empty=image is not None)
        if candidate is None:
            return
        self._conv.commit(candidate)
        if image is not None:  # the screen goes with this answer only; the next one starts clean
            self._shot = None
            self._shot_armed = False
            self._ui.screenshot_state.emit("idle")
        job, self._spec = self._spec, None
        if job is not None and job.answer_id is None:
            if job.candidate.key == candidate.key and job.image is image:
                log.info("speculative answer reused")
                self._show(job, t_pause)
                return
            job.task.cancel()  # type: ignore[union-attr]
        job = self._start_job(candidate, image=image)
        self._show(job, t_pause)

    # ---- manual term correction (PAUSED only) ---------------------------------------------

    def retry(self, note: str) -> None:
        """Re-answer the question already on screen, following a note typed by hand.

        Called from the Qt thread via call_soon_threadsafe, like toggle().
        """
        note = note.strip()
        if not note:
            return
        if self._listening or self._last is None:
            self._ui.notice.emit("Önce Space ile duraklat ve bir cevap al, sonra notu yaz.")
            return
        prev = self._retry
        if (prev is not None and prev.note == note and prev.provider is None and prev.error is None
                and self._shot is None):
            return  # same correction already streaming (double Enter)
        # The screen of that answer stays; a picture taken after it (while paused) replaces it.
        image = self._shot or self._last.image
        if self._shot is not None:
            self._shot = None
            self._ui.screenshot_state.emit("idle")
        log.info("correcting with note %r%s", note, " + screen" if image else "")
        # The reply on screen was wrong: the model must see it, or it gives the same reading again.
        wrong = " ".join("".join(self._last.chunks).split())[:REJECTED_CHARS]
        rejected = (*self._last.rejected, wrong)[-REJECTED_KEPT:] if wrong else self._last.rejected
        self._retry = self._start_job(
            self._last.candidate, note=note, retry=True, image=image, rejected=rejected
        )
        self._show(self._retry, time.monotonic())

    # ---- answer jobs --------------------------------------------------------------------

    def _start_job(
        self, candidate: Candidate, *, note: str = "", retry: bool = False, image: Shot | None = None,
        rejected: tuple[str, ...] = (),
    ) -> _Job:
        job = _Job(candidate, note, retry, image, rejected)
        job.task = self._spawn(self._run_job(job))
        return job

    async def _run_job(self, job: _Job) -> None:
        def on_delta(text: str) -> None:
            if job.first_token_at is None:
                job.first_token_at = time.monotonic()
            job.chunks.append(text)
            if job.answer_id is not None:
                self._ui.answer_delta.emit(job.answer_id, text)

        def on_reset() -> None:
            job.chunks.clear()
            job.first_token_at = None
            if job.answer_id is not None:
                self._ui.answer_reset.emit(job.answer_id)

        c = job.candidate
        try:
            job.provider = await self._engine.answer(
                c.question, c.history, on_delta, on_reset, note=job.note, image=job.image,
                rejected=job.rejected,
            )
        except AnswerError as exc:
            job.error = str(exc)
        if job.answer_id is not None:
            self._finish(job)

    def _show(self, job: _Job, t_pause: float) -> None:
        self._answer_seq += 1
        job.answer_id = self._answer_seq
        job.shown_at = t_pause
        self._last = job
        note = job.note if len(job.note) <= CARD_NOTE_CHARS else job.note[: CARD_NOTE_CHARS - 1] + "…"
        question = job.candidate.question or "(konuşma yok)"
        title = f"{question}  ·  \u21bb {note}" if note else question
        if job.image is not None:
            title = "\U0001f4f7 " + title
        self._ui.answer_started.emit(job.answer_id, title)
        if job.chunks:
            self._ui.answer_delta.emit(job.answer_id, "".join(job.chunks))
        if job.provider is not None or job.error is not None:
            self._finish(job)

    def _finish(self, job: _Job) -> None:
        assert job.answer_id is not None
        if job.error is not None:
            self._ui.answer_failed.emit(job.answer_id, job.error)
            return
        # Read aloud, so it joins the interview history; a correction replaces the wrong answer.
        text = "".join(job.chunks).strip()
        self._conv.record_answer(job.candidate, text, replace=job.retry, append=job.image is not None)
        screen = job.image.file if job.image is not None else ""
        self._record.answer(job.answer_id, job.candidate.question, text, job.provider or "", job.note, screen)
        wait = max(0.0, (job.first_token_at or time.monotonic()) - job.shown_at)
        log.info("answer #%d via %s, first token %.2fs after pause", job.answer_id, job.provider, wait)
        self._ui.answer_done.emit(job.answer_id, f"{job.provider} · {'hazırdı' if wait < 0.05 else f'{wait:.1f}s'}")
