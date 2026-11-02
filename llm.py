"""Answer generation: Claude Sonnet 5 streaming (thinking off by default, low/medium/high selectable
in the UI), Haiku 4.5 / Gemini as fast backups."""
from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import anthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from config import (
    CACHE_REFRESH_S,
    CLAUDE_COOLDOWN_S,
    CLAUDE_DISABLE_S,
    FIRST_TOKEN_TIMEOUT_S,
    HEDGE_AFTER_S,
    MAX_TOKENS_CLAUDE,
    MAX_TOKENS_GEMINI,
    REQUEST_TIMEOUT_S,
    THINKING_LEVELS,
    THINKING_OFF,
    WARM_IDLE_S,
    Config,
    anthropic_headers,
)
from conversation import INTERVIEWER, Turn
from languages import english_name

log = logging.getLogger(__name__)

# Used only if prompt.txt is missing or empty; the real instructions live in prompt.txt.
_FALLBACK_PROMPT = (
    "You are the candidate described in <cv>, in a live English job interview for the job in <job>. "
    "Reply to what the interviewer is doing in their last words, as a short, natural B1 spoken reply, "
    "using only facts from <cv>, <job> and the conversation. Output only the spoken words."
)

_FALLBACK_PROMPT_TR = (
    "Sen <cv>'de anlatılan adaysın, <job>'daki iş için canlı bir Türkçe iş görüşmesindesin. "
    "Görüşmecinin son sözlerinde ne yaptığına kısa, doğal bir konuşma cevabıyla karşılık ver, "
    "sadece <cv>, <job> ve konuşmadaki gerçekleri kullan. Sadece söyleyeceğin kelimeleri çıktı olarak ver."
)

# Lowest thinking first; a model that rejects a level gets the next one.
_GEMINI_THINKING = (types.ThinkingLevel.MINIMAL, types.ThinkingLevel.LOW, None)


class AnswerError(Exception):
    pass


class _StreamReset(Exception):
    """Provider failed after emitting text; the UI must clear the partial answer."""


_REMINDER = (
    "From the app, not said by the interviewer. First decide what they are doing in their last words, then "
    "give the reply that moment needs. An explanation or a statement gets a short reaction in one or two "
    "sentences, without retelling it and without an example from <cv>. A greeting or a check gets a few words. "
    "A question about you or your opinion gets the answer first and one concrete fact, in two to four short "
    "sentences under 60 words; tell me about yourself, why us and strengths at most five short sentences, "
    "never over 80 words, naming internships and projects without details. A technical question gets a correct, complete "
    "answer: the direct answer first, then every point a good answer needs, each said once in a short "
    "sentence, never cut short to save words and never padded. A coding request gets a short spoken analysis (idea, Big O, an edge case), then the "
    "simplest Python code on its own lines with four-space indentation and no ``` marks. Use what they "
    "already told you, don't repeat an example, a phrase or an opening you used earlier, and don't invent "
    "anything about you beyond <cv>, <job> and this conversation: no anecdote, conflict, failure, number, "
    "district or result that <cv> does not state, and never say you have no conflict or failure to tell; "
    "say how you handle it. Sound like the candidate in <cv> talking: easy B1 sentences in one "
    "paragraph, except code, only the words they say."
)

_REMINDER_TR = (
    "Uygulamadan, görüşmeci söylemedi. Önce son sözlerinde ne yaptıklarına karar ver, sonra o anın "
    "gerektirdiği cevabı ver. Bir açıklama veya ifade bir iki cümlelik kısa bir tepki alır, anlattıklarını "
    "özetlemeden ve <cv>'den örnek vermeden. Selamlaşma veya kontrol birkaç kelime alır. Senin hakkında bir "
    "soru veya fikir sorusu önce cevap ve tek bir somut bilgiyle iki ila dört kısa cümle alır, 50 kelimenin altında; "
    "kendinizi anlatın, neden biz ve güçlü yönler en fazla beş kısa cümle, asla 70 kelimeden fazla değil, staj "
    "ve projelerin adı geçer ama ayrıntısı geçmez. Teknik soru doğru ve eksiksiz bir cevap alır: önce doğrudan cevap, "
    "sonra iyi bir cevapta olması gereken her nokta, her biri bir kez ve kısa bir cümleyle; kelime kazanmak "
    "için asla kısaltma, dolgu da ekleme, cümleleri virgülle uzatma. "
    "Kod isteği önce kısa, konuşur gibi bir analiz (fikir, Big O, bir uç durum), sonra kendi satırlarında, "
    "dört boşluk girintili ve ``` işareti olmadan en basit Python kodunu alır. Söylediklerini kullan, daha "
    "önce kullandığın örneği, ifadeyi veya açılışı tekrarlama, senin hakkında <cv>, <job> ve bu konuşmanın "
    "dışında bir şey uydurma: <cv>'de yazmayan anekdot, anlaşmazlık, başarısızlık, sayı, semt veya sonuç yok; "
    "anlaşmazlık veya başarısızlık örneğin olmadığını da asla söyleme, nasıl davrandığını anlat. "
    "Konuşmanın kesik gelmesini asla yorumlama. <cv>'deki aday konuşuyormuş gibi doğal, sade Türkçe, görüşmeciye "
    "\"siz\" diye, kod hariç tek paragraf, sadece söyleyeceği kelimeler."
)

_OPENINGS_CHECKED = 8     # recent replies whose first words are listed, so openers don't repeat
_OPENING_WORDS = 4


# Sent only on a manual correction. Free text typed by the candidate, so it is placed last,
# right before the reminder, where the model follows it best.
_NOTE = (
    "<note_from_you>You typed this by hand on the correction bar, because the reply you just gave was wrong "
    "or missed the point. Nobody said it out loud. It may be in Turkish or English; understand it either way "
    "and still reply in spoken English.\n"
    "Note: {note}\n"
    "It outranks the speech above. It can be a word the speech-to-text garbled, like \"rdap\", and then that "
    "word is what they asked about. Or it can be a direction, like \"rdap ile ilgili bir örnek ver\" (give an "
    "example about RDAP) or \"daha kısa\" (shorter), and then you do exactly that. Don't repeat the previous "
    "reply, and never read out, translate or mention the note, the transcript or a hearing problem.</note_from_you>"
)

_NOTE_TR = (
    "<senin_notun>Bunu düzeltme çubuğuna elle yazdın, çünkü az önce verdiğin cevap yanlıştı "
    "veya amacı kaçırdı. Kimse bunu sesli söylemedi. Türkçe veya İngilizce olabilir; her iki şekilde de anla "
    "ve yine Türkçe konuşarak cevap ver.\n"
    "Not: {note}\n"
    "Yukarıdaki konuşmayı geçersiz kılar. Konuşma tanımanın bozduğu bir kelime olabilir, mesela \"rdap\", o zaman "
    "sordukları kelime odur. Veya bir yönlendirme olabilir, mesela \"rdap ile ilgili bir örnek ver\" veya "
    "\"daha kısa\", o zaman tam olarak bunu yap. Önceki cevabı tekrarlama; notu, transkripti "
    "veya duyma sorununu asla okuma, çevirme veya bunlardan bahsetme.</senin_notun>"
)


@dataclass(frozen=True)
class _Prompt:
    history: Sequence[Turn]   # earlier exchanges, starting with the interviewer and ending with you
    final: str                # the interviewer's newest speech plus the reminder


StreamFn = Callable[[_Prompt, Callable[[str], None]], Awaitable[None]]


def _openings(history: Sequence[Turn]) -> list[str]:
    """First words of your recent replies, e.g. "That makes sense"; the model rarely tracks these itself."""
    said = [t.text for t in history if t.role != INTERVIEWER][-_OPENINGS_CHECKED:]
    found = []
    for text in said:
        first_clause = re.split(r"[.,!?]", text, maxsplit=1)[0]
        if words := first_clause.split()[:_OPENING_WORDS]:
            found.append(" ".join(words))
    return list(dict.fromkeys(found))


_THINK = (
    " Before you reply, think it through: what kind of moment this is and the right length, which facts "
    "from <cv> and <job> fit, and check that every fact you will say is really in <cv>, <job> or this "
    "conversation."
)
_THINK_TR = (
    " Cevaplamadan önce düşün: bu nasıl bir an ve doğru uzunluk ne, <cv> ve <job>'dan hangi bilgiler uyuyor, "
    "ve söyleyeceğin her bilginin gerçekten <cv>, <job> veya bu konuşmada olduğunu kontrol et."
)


def _language_rule(lang: str) -> str:
    """Added to the English prompt for languages without their own prompt file."""
    name = english_name(lang)
    return (
        f"\n\nINTERVIEW LANGUAGE\n\nThis interview is held in {name}, not English. Everything above still "
        f"applies, but wherever it says English, read {name}: the interviewer speaks {name}, the transcript is "
        f"{name} speech-to-text, and you always reply in natural, simple spoken {name}, at the same length and "
        f"level. <cv> and <job> may be in English; still reply only in {name}. Keep technical terms and product "
        "names as people in that language say them."
    )


def build_final_message(
    question: str, history: Sequence[Turn] = (), note: str = "", lang: str = "en", think: bool = False
) -> str:
    text = question
    note_tpl = _NOTE_TR if lang == "tr" else _NOTE
    if lang not in ("en", "tr"):
        note_tpl = note_tpl.replace("spoken English", f"spoken {english_name(lang)}")
    if note := note.strip():
        text += "\n\n" + note_tpl.format(note=note)
    reminder = _REMINDER_TR if lang == "tr" else _REMINDER
    if think:  # adaptive thinking skips short chat turns unless asked to think
        reminder += _THINK_TR if lang == "tr" else _THINK
    if openings := _openings(history):
        listed = "; ".join(f'"{o}"' for o in openings)
        if lang == "tr":
            reminder += f" Son cevapların şöyle başladı: {listed}, bu sefer başka bir şekilde başla."
        else:
            reminder += f" Your recent replies began with {listed}, so begin this one another way."
    if lang not in ("en", "tr"):
        reminder += f" Reply only in {english_name(lang)}."
    tag = "hatırlatma" if lang == "tr" else "reminder"
    return f"{text}\n\n<{tag}>{reminder}</{tag}>"


def claude_messages(history: Sequence[Turn], final: str) -> list[dict]:
    """The interview as a real dialogue; the cache breakpoint sits on the last turn already said.

    That prefix is identical for every request until the next answer, so speculative requests and the
    answer itself read the whole interview from cache and only pay full price for the newest speech.
    """
    msgs: list[dict] = [
        {"role": "user" if t.role == INTERVIEWER else "assistant", "content": t.text} for t in history
    ]
    if msgs:
        last = msgs[-1]
        last["content"] = [{"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}]
    msgs.append({"role": "user", "content": final})
    return msgs


def gemini_contents(history: Sequence[Turn], final: str) -> list[types.Content]:
    contents = [
        types.Content(role="user" if t.role == INTERVIEWER else "model", parts=[types.Part(text=t.text)])
        for t in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=final)]))
    return contents


class AnswerEngine:
    """Streams answers from the primary Claude model, with a fast backup raced only when needed.

    Order: primary Claude model (Sonnet 5) -> backup Claude model (Haiku 4.5) -> Gemini.
    The next provider starts only if the current one fails or has not produced a token
    after HEDGE_AFTER_S, so a normal answer costs exactly one request.
    """

    def __init__(self, cfg: Config, on_provider: Callable[[str], None]) -> None:
        self._cfg = cfg
        self._lang = "en"
        self._thinking = THINKING_OFF
        system = self._build_system(cfg.prompt_text, cfg.job_text, cfg.cv_text)
        system_tr = self._build_system(
            cfg.prompt_text_tr or _FALLBACK_PROMPT_TR,
            cfg.job_text_tr, cfg.cv_text_tr or cfg.cv_text,
            fallback=_FALLBACK_PROMPT_TR,
        )
        self._system = system
        self._system_tr = system_tr
        self._active_system = system
        self._other_system = cfg.prompt_text or _FALLBACK_PROMPT  # base for languages without own prompt files
        self._job_cv = (cfg.job_text, cfg.cv_text)
        # Cached prefix: Sonnet 5 caches prompts from 1,024 tokens, so every repeat is ~10% input price.
        self._claude_system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self._on_provider = on_provider
        self._provider_label = ""
        self._last_activity = 0.0
        self._cache_used_at = 0.0  # last request that read or wrote the primary model's prompt cache
        self._warm_tasks: set[asyncio.Task[None]] = set()

        self._claude: anthropic.AsyncAnthropic | None = None
        self._claude_models: list[str] = []
        if cfg.anthropic_key:
            self._claude = anthropic.AsyncAnthropic(
                api_key=cfg.anthropic_key,
                max_retries=0,  # the backup provider takes over instead of retrying
                timeout=REQUEST_TIMEOUT_S,
                http_client=_claude_http_client(),
                default_headers=anthropic_headers(cfg),
            )
            self._claude_models = [m for m in (cfg.claude_model, cfg.claude_backup_model) if m]
            self._claude_models = list(dict.fromkeys(self._claude_models))  # drop duplicates
        self._blocked_until: dict[str, float] = {}

        self._gemini: genai.Client | None = None
        self._gemini_model = cfg.gemini_model
        if cfg.gemini_key:
            self._gemini = genai.Client(
                api_key=cfg.gemini_key,
                http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_S * 1000)),
            )
        self._gemini_thinking = 0
        self._gemini_config = self._make_gemini_config()

    @staticmethod
    def _build_system(prompt_text: str, job_text: str, cv_text: str, fallback: str = "") -> str:
        system = f"{prompt_text or fallback or _FALLBACK_PROMPT}\n\nToday is {time.strftime('%d %B %Y')}."
        if job_text:
            system += f"\n\n<job>\n{job_text}\n</job>"
        if cv_text:
            system += f"\n\n<cv>\n{cv_text}\n</cv>"
        return system

    def set_language(self, lang: str) -> None:
        """Switch the active system prompt: own files for English and Turkish, the English prompt plus a
        language rule for every other language."""
        self._lang = lang
        if lang == "tr":
            self._active_system = self._system_tr
        elif lang == "en":
            self._active_system = self._system
        else:
            self._active_system = self._build_system(self._other_system + _language_rule(lang), *self._job_cv)
        self._claude_system = [{"type": "text", "text": self._active_system, "cache_control": {"type": "ephemeral"}}]
        self._cache_used_at = 0.0  # force cache refresh on next warm()
        self._gemini_config = self._make_gemini_config()
        log.info("LLM language set to %s", lang)

    def set_thinking(self, label: str) -> None:
        """Thinking depth of the primary model: off (fastest) or low / medium / high effort."""
        if label not in THINKING_LEVELS or label == self._thinking:
            return
        self._thinking = label
        self._cache_used_at = 0.0  # thinking settings are part of the cache key: re-warm with the new ones
        log.info("primary thinking -> %s", label)

    def primary_ready(self) -> bool:
        """The primary Claude model can answer now (thinking settings only apply to it)."""
        ready = self._claude_ready()
        return bool(ready) and ready[0] == self._claude_models[0]

    def _level(self) -> tuple[str | None, int, float, float]:
        return THINKING_LEVELS[self._thinking]

    def _claude_params(self, model: str) -> dict:
        """Thinking settings: only the primary model thinks; the backups always answer at once."""
        effort = self._level()[0] if model == self._claude_models[0] else None
        if effort is None:
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}

    def _make_gemini_config(self) -> types.GenerateContentConfig:
        if self._gemini_model.startswith("gemini-2.5"):
            thinking = types.ThinkingConfig(thinking_budget=0)  # 2.5 models: budget 0 = no thinking
        else:
            level = _GEMINI_THINKING[self._gemini_thinking]
            thinking = types.ThinkingConfig(thinking_level=level) if level else None
        return types.GenerateContentConfig(
            system_instruction=self._active_system,
            max_output_tokens=MAX_TOKENS_GEMINI,
            thinking_config=thinking,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    # ---- provider state -----------------------------------------------------------

    def _available(self, model: str) -> bool:
        return time.monotonic() >= self._blocked_until.get(model, 0.0)

    def _claude_ready(self) -> list[str]:
        return [m for m in self._claude_models if self._available(m)] if self._claude else []

    def active_provider(self) -> str:
        ready = [_label(m) for m in self._claude_ready()]
        if self._gemini is not None:
            ready.append("Gemini")
        if not ready:
            return "LLM yok"
        return ready[0] + (f" (yedek: {', '.join(ready[1:])})" if len(ready) > 1 else "")

    def _emit_provider(self) -> None:
        label = self.active_provider()
        if label != self._provider_label:
            self._provider_label = label
            self._on_provider(label)

    def _block(self, models: Sequence[str], seconds: float, reason: str) -> None:
        until = time.monotonic() + seconds
        for m in models:
            self._blocked_until[m] = until
        log.warning("%s paused for %.0fs: %s", ", ".join(models), seconds, reason)
        self._emit_provider()

    def _classify_claude_error(self, model: str, exc: BaseException) -> None:
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            self._block(self._claude_models, CLAUDE_DISABLE_S, "auth")  # same key for all models
        elif isinstance(exc, anthropic.BadRequestError) and "credit" in str(exc).lower():
            self._block(self._claude_models, CLAUDE_DISABLE_S, "billing")
        elif isinstance(exc, anthropic.BadRequestError) and "workspace" in str(exc).lower():
            self._block(self._claude_models, CLAUDE_DISABLE_S, "key needs ANTHROPIC_WORKSPACE_ID")
        elif isinstance(exc, anthropic.NotFoundError):
            self._block([model], CLAUDE_DISABLE_S, "model not found")
        elif isinstance(exc, (anthropic.RateLimitError, anthropic.InternalServerError)):
            self._block([model], CLAUDE_COOLDOWN_S, type(exc).__name__)  # limits are per model
        elif isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
            self._block([model], CLAUDE_COOLDOWN_S, f"HTTP {exc.status_code}")

    # ---- startup check / connection warm-up --------------------------------------

    async def check(self) -> list[str]:
        """Validate keys/models once at startup (also opens the HTTPS connections)."""
        problems: list[str] = []
        for i, model in enumerate(self._claude_models):
            try:
                if i == 0:
                    await self._prewarm(model)  # the first answer then reads the prompt from cache
                else:
                    await self._claude.models.retrieve(model)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001 - the next provider takes over
                log.error("%s check failed: %r", model, exc)
                self._classify_claude_error(model, exc)
                problems.append(f"{_label(model)} kullanılamıyor: {_short(exc)}")
        if self._gemini is not None:
            try:
                await self._gemini.aio.models.get(model=self._gemini_model)
            except Exception as exc:  # noqa: BLE001
                log.error("gemini check failed: %r", exc)
                problems.append(f"Gemini yedeği kullanılamıyor: {_short(exc)}")
        self._last_activity = time.monotonic()
        self._emit_provider()
        return problems

    def warm(self, history: Sequence[Turn] = ()) -> None:
        """Called while the interviewer speaks, before the question is complete.

        Refreshes the prompt cache (rules, CV and the interview so far) if it is about to expire,
        otherwise just keeps the HTTPS connection open (free), so the answer itself starts fast.
        """
        ready = self._claude_ready()
        if not ready or ready[0] != self._claude_models[0]:
            return
        now = time.monotonic()
        if now - self._cache_used_at > CACHE_REFRESH_S:
            coro = self._prewarm(ready[0], history)
        elif now - self._last_activity > WARM_IDLE_S:
            coro = self._claude.models.retrieve(ready[0])  # type: ignore[union-attr]
        else:
            return
        self._last_activity = now
        task = asyncio.get_running_loop().create_task(self._best_effort(coro))
        self._warm_tasks.add(task)
        task.add_done_callback(self._warm_tasks.discard)

    @staticmethod
    async def _best_effort(coro: Awaitable[object]) -> None:
        try:
            await coro
        except Exception as exc:  # noqa: BLE001 - warm-up must never disturb the interview
            log.debug("warm-up failed: %r", exc)

    async def _prewarm(self, model: str, history: Sequence[Turn] = ()) -> None:
        """Write/refresh the cached prefix without generating anything (max_tokens=0).

        The placeholder "." comes after the last breakpoint, so the cache matches real requests.
        """
        self._cache_used_at = time.monotonic()
        msg = await self._claude.messages.create(  # type: ignore[union-attr]
            model=model,
            max_tokens=0,
            system=self._claude_system,
            messages=claude_messages(history, "."),
            **self._claude_params(model),  # must match real requests to share the cache
        )
        u = msg.usage
        log.info("%s cache warm: read=%s write=%s", _label(model), u.cache_read_input_tokens, u.cache_creation_input_tokens)

    # ---- generation ---------------------------------------------------------------

    async def answer(
        self,
        question: str,
        history: Sequence[Turn],
        on_delta: Callable[[str], None],
        on_reset: Callable[[], None],
        note: str = "",
    ) -> str:
        """Stream an answer and return the label of the provider that produced it."""
        thinks = self._level()[0] is not None and self.primary_ready()
        prompt = _Prompt(tuple(history), build_final_message(question, history, note, lang=self._lang, think=thinks))
        self._last_activity = time.monotonic()
        self._emit_provider()  # e.g. back to the primary model after a cooldown

        providers: list[tuple[str, str | None, StreamFn]] = [
            (_label(m), m, functools.partial(self._stream_claude, m)) for m in self._claude_ready()
        ]
        if self._gemini is not None:
            providers.append(("Gemini", None, self._stream_gemini))
        if not providers:
            raise AnswerError("Kullanılabilir model yok (anahtarları kontrol et)")

        winner: str | None = None
        got_token = asyncio.Event()
        tasks: dict[str, asyncio.Task[None]] = {}
        failures: dict[str, BaseException] = {}

        def deliver_for(label: str) -> Callable[[str], None]:
            def deliver(text: str) -> None:
                nonlocal winner
                if winner is None:
                    winner = label
                    got_token.set()
                    for other, task in tasks.items():
                        if other != label:
                            task.cancel()  # the slower request is dropped immediately
                if winner == label:
                    on_delta(text)

            return deliver

        async def attempt(label: str, model: str | None, stream: StreamFn, sink: Callable[[str], None]) -> None:
            try:
                await stream(prompt, sink)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, the next provider takes over
                cause = exc.__cause__ if isinstance(exc, _StreamReset) and exc.__cause__ else exc
                failures[label] = cause
                log.warning("%s failed: %r", label, cause)
                if model is not None:
                    self._classify_claude_error(model, cause)

        def launch(label: str, model: str | None, stream: StreamFn, sink: Callable[[str], None] | None = None) -> None:
            coro = attempt(label, model, stream, sink or deliver_for(label))
            tasks[label] = asyncio.get_running_loop().create_task(coro)

        def usable(model: str | None) -> bool:
            return model is None or self._available(model)  # skip a model that just hit a limit

        backups = providers[1:]
        # A thinking primary is silent while it thinks, so the backup waits longer before racing it.
        primary_thinks = bool(self._claude_ready()) and self._claude_ready()[0] == self._claude_models[0]
        hedge_after = self._level()[3] if primary_thinks else HEDGE_AFTER_S
        launch(*providers[0])
        try:
            while winner is None:
                running = [t for t in tasks.values() if not t.done()]
                if not running:
                    while backups and not usable(backups[0][1]):
                        backups.pop(0)
                    if not backups:
                        break
                    launch(*backups.pop(0))  # the current provider failed: next one right away
                    continue
                token_wait = asyncio.get_running_loop().create_task(got_token.wait())
                try:
                    done, _ = await asyncio.wait(
                        [*running, token_wait],
                        timeout=hedge_after if backups else None,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    token_wait.cancel()  # also when this answer is cancelled, so no waiter is left pending
                if not done:
                    while backups and not usable(backups[0][1]):
                        backups.pop(0)
                    if backups:
                        log.info("no token after %.1fs, racing %s", hedge_after, backups[0][0])
                        launch(*backups.pop(0))

            if winner is not None:
                await tasks[winner]
                if winner not in failures:
                    return winner
                on_reset()  # the winner broke mid-stream: try the others one by one
                for label, model, stream in providers:
                    if label in failures or label == winner or not usable(model):
                        continue
                    launch(label, model, stream, on_delta)
                    await tasks[label]
                    if label not in failures:
                        return label
                    on_reset()
        finally:
            for task in tasks.values():
                task.cancel()

        last = next(reversed(failures.values()), None)
        raise AnswerError(f"Cevap alınamadı: {_short(last)}")

    async def _stream_claude(self, model: str, prompt: _Prompt, on_delta: Callable[[str], None]) -> None:
        assert self._claude is not None
        if model == self._claude_models[0]:
            self._cache_used_at = time.monotonic()  # cache lifetime counts from request start
        emitted = False
        params = self._claude_params(model)
        if "output_config" in params:  # thinking: more room and time before the first visible word
            _, max_tokens, first_token_s, _ = self._level()
            client = self._claude.with_options(timeout=first_token_s + REQUEST_TIMEOUT_S)
        else:
            max_tokens, first_token_s, client = MAX_TOKENS_CLAUDE, FIRST_TOKEN_TIMEOUT_S, self._claude
        try:
            async with asyncio.timeout(first_token_s) as deadline:
                async with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=self._claude_system,
                    messages=claude_messages(prompt.history, prompt.final),
                    **params,  # off by default: Sonnet 5 thinks unless told not to, and speed matters
                ) as stream:
                    async for text in stream.text_stream:
                        if not emitted:
                            emitted = True
                            deadline.reschedule(None)
                        on_delta(text)
                    usage = (await stream.get_final_message()).usage
        except Exception as exc:
            if emitted:
                raise _StreamReset from exc
            raise
        if not emitted:
            raise AnswerError(f"{_label(model)} boş cevap döndü")
        log.info(
            "%s tokens: cache_read=%s cache_write=%s uncached_in=%s out=%s thinking=%s",
            _label(model), usage.cache_read_input_tokens, usage.cache_creation_input_tokens,
            usage.input_tokens, usage.output_tokens, params.get("output_config", {}).get("effort", "off"),
        )

    async def _stream_gemini(self, prompt: _Prompt, on_delta: Callable[[str], None]) -> None:
        while True:
            try:
                await self._stream_gemini_once(prompt, on_delta)
                return
            except genai_errors.ClientError as exc:
                # Some models reject a thinking level; step to the next one and retry.
                last = self._gemini_thinking >= len(_GEMINI_THINKING) - 1
                if exc.code != 400 or "thinking" not in str(exc).lower() or last:
                    raise
                self._gemini_thinking += 1
                log.warning("gemini rejected thinking level, trying %s", _GEMINI_THINKING[self._gemini_thinking])
                self._gemini_config = self._make_gemini_config()

    async def _stream_gemini_once(self, prompt: _Prompt, on_delta: Callable[[str], None]) -> None:
        assert self._gemini is not None
        emitted = False
        try:
            async with asyncio.timeout(FIRST_TOKEN_TIMEOUT_S + 3) as deadline:
                stream = await self._gemini.aio.models.generate_content_stream(
                    model=self._gemini_model,
                    contents=gemini_contents(prompt.history, prompt.final),
                    config=self._gemini_config,
                )
                async for chunk in stream:
                    text = chunk.text
                    if not text:
                        continue
                    if not emitted:
                        emitted = True
                        deadline.reschedule(None)
                    on_delta(text)
        except Exception as exc:
            if emitted:
                raise _StreamReset from exc
            raise
        if not emitted:
            raise AnswerError("Gemini boş cevap döndü")

    async def close(self) -> None:
        if self._claude is not None:
            await self._claude.close()
        if self._gemini is not None:
            try:
                await self._gemini.aio.aclose()
            except Exception:  # noqa: BLE001
                pass


def _label(model: str) -> str:
    """Short display name: claude-sonnet-5 -> Sonnet 5, claude-haiku-4-5-20251001 -> Haiku 4.5."""
    parts = model.removeprefix("claude-").split("-")
    if not parts or not parts[0].isalpha():
        return model
    version = [p for p in parts[1:] if p.isdigit() and len(p) < 3]
    return f"{parts[0].capitalize()} {'.'.join(version)}".strip()


def _claude_http_client() -> anthropic.DefaultAsyncHttpxClient | None:
    """Longer keep-alive than the SDK's 5 s so a warmed connection survives until the question."""
    try:
        import httpx2  # transport used by anthropic>=1.x

        return anthropic.DefaultAsyncHttpxClient(
            limits=httpx2.Limits(max_connections=20, max_keepalive_connections=5, keepalive_expiry=90.0)
        )
    except Exception:  # noqa: BLE001 - fall back to SDK defaults
        return None


def _short(exc: BaseException | None) -> str:
    """Human-readable one-liner for the UI."""
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "API anahtarı geçersiz"
    if isinstance(exc, anthropic.RateLimitError):
        return "limit doldu (429)"
    if isinstance(exc, anthropic.NotFoundError):
        return "model bulunamadı"
    if isinstance(exc, anthropic.BadRequestError) and "workspace" in str(exc).lower():
        return "anahtar workspace'e bağlı değil (.env: ANTHROPIC_WORKSPACE_ID)"
    if isinstance(exc, anthropic.APIStatusError):
        text = f"HTTP {exc.status_code}"
    elif isinstance(exc, genai_errors.APIError):
        text = f"{exc.code} {exc.message or exc.status or ''}".strip()
    elif isinstance(exc, (TimeoutError, anthropic.APITimeoutError)):
        return "zaman aşımı"
    elif isinstance(exc, anthropic.APIConnectionError):
        return "bağlantı hatası"
    else:
        text = str(exc) or type(exc).__name__
    return text if len(text) <= 90 else text[:87] + "…"
