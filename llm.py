"""Answer generation: Claude Sonnet 5 streaming (thinking off by default, low/medium/high selectable
in the UI), Haiku 4.5 / Gemini as fast backups."""
from __future__ import annotations

import asyncio
import base64
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
    GEMINI,
    MAX_TOKENS_CLAUDE,
    MAX_TOKENS_GEMINI,
    MODEL_CHOICES,
    NO_ADAPTIVE_THINKING,
    REQUEST_TIMEOUT_S,
    SHOT_EXTRA_WAIT_S,
    SHOT_MAX_TOKENS_CLAUDE,
    THINKING_LEVELS,
    THINKING_OFF,
    THINKING_OFF_PARAMS,
    WARM_IDLE_S,
    Config,
    anthropic_headers,
)
from conversation import INTERVIEWER, Turn
from languages import english_name
from screenshot import Shot

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
    "never over 80 words, naming internships and projects without details. A technical question gets a correct, solution-focused "
    "answer: the direct answer first, then only the points that matter, each said once in a short sentence, "
    "usually 40 to 80 words, no intro and no background. How you would approach or handle something gets your decision "
    "first, then two or three steps, about 40 to 70 words. How you would build or design a system gets the "
    "SYSTEM DESIGN answer, about 80 to 120 words: the first sentence is already the architecture with concrete "
    "parts, never a framing or a restated problem; then the data and labeling, why this over the simpler "
    "option, the pipeline with tool names, and how you evaluate it against human labels; production or a risk "
    "only if their case needs it, in one short clause. Every sentence short with one idea, no purpose tails "
    "like \"so that…\" or \"to lower cost\", no labels, never reuse the sentences of the example. A coding request gets a short spoken analysis (idea, Big O, an edge case), then the "
    "simplest Python code on its own lines with four-space indentation and no ``` marks. Use what they "
    "already told you, don't repeat an example, a phrase or an opening you used earlier, and don't invent "
    "anything about you beyond <cv>, <job> and this conversation: no anecdote, conflict, failure, number, "
    "district or result that <cv> does not state, and never say you have no conflict or failure to tell; "
    "say how you handle it. Give a reason only for the main choice, never end with a \"So, …\" summary. "
    "Sound like the candidate in <cv> talking: easy B1 sentences in one "
    "paragraph, except code, only the words they say."
)

_REMINDER_TR = (
    "Uygulamadan, görüşmeci söylemedi. Önce son sözlerinde ne yaptıklarına karar ver, sonra o anın "
    "gerektirdiği cevabı ver. Bir açıklama veya ifade bir iki cümlelik kısa bir tepki alır, anlattıklarını "
    "özetlemeden ve <cv>'den örnek vermeden. Selamlaşma veya kontrol birkaç kelime alır. Senin hakkında bir "
    "soru veya fikir sorusu önce cevap ve tek bir somut bilgiyle iki ila dört kısa cümle alır, 50 kelimenin altında; "
    "kendinizi anlatın, neden biz ve güçlü yönler en fazla beş kısa cümle, asla 70 kelimeden fazla değil, staj "
    "ve projelerin adı geçer ama ayrıntısı geçmez. Teknik soru doğru ve çözüm odaklı bir cevap alır: önce doğrudan cevap, "
    "sonra sadece önemli noktalar, her biri bir kez ve kısa bir cümleyle, genelde 35 ila 70 kelime, giriş ve arka plan "
    "yok, dolgu yok, cümleleri virgülle uzatma. Bir duruma nasıl yaklaşacağın önce kararını, sonra iki üç "
    "adımı alır, yaklaşık 35 ila 60 kelime. Bir sistemi nasıl kuracağın veya "
    "tasarlayacağın SİSTEM TASARIMI cevabını alır, yaklaşık 70 ila 100 kelime: ilk cümle zaten somut "
    "parçalarıyla mimaridir, giriş veya problemi tekrar yok; sonra veri ve etiketleme, daha basit seçeneğe "
    "karşı neden bu, araç adlarıyla pipeline ve insan etiketlerine karşı değerlendirme; canlı ortam veya risk "
    "sadece durumu gerektiriyorsa, kısa bir yan cümleyle. Her cümle kısa ve tek fikir, \"… diye\", \"… için\" "
    "gibi amaç ekleri ve \"yani\" yok, başlık başlık sayma yok, örnekteki cümleleri tekrar kullanma. "
    "Kod isteği önce kısa, konuşur gibi bir analiz (fikir, Big O, bir uç durum), sonra kendi satırlarında, "
    "dört boşluk girintili ve ``` işareti olmadan en basit Python kodunu alır. Söylediklerini kullan, daha "
    "önce kullandığın örneği, ifadeyi veya açılışı tekrarlama, senin hakkında <cv>, <job> ve bu konuşmanın "
    "dışında bir şey uydurma: <cv>'de yazmayan anekdot, anlaşmazlık, başarısızlık, sayı, semt veya sonuç yok; "
    "anlaşmazlık veya başarısızlık örneğin olmadığını da asla söyleme, nasıl davrandığını anlat. "
    "Sadece ana tercihin sebebini söyle, asla \"Yani…\" diye özetle bitirme. "
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
    "Note: {note}\n{rejected}"
    "It outranks the speech above. It can be the word the speech-to-text garbled, like \"rdap\": then that "
    "exact word is what they asked about. Take it as written and never swap it for a better-known word that "
    "looks or sounds alike. A hint next to it, like \"ai model\" or \"kütüphane\", tells you its kind: pick "
    "the real term of that kind that is closest to the word. If you are not "
    "sure such a term exists and what it does, never describe it from its name; say in one sentence that you haven't used it and explain the closest thing you know; "
    "never invent what it does. The note can also be a direction, like \"rdap ile ilgili bir örnek ver\" (give "
    "an example about RDAP) or \"daha kısa\" (shorter), and then you do exactly that. Never give the same "
    "topic or reading as a wrong reply again, and never read out, translate or mention the note, the "
    "transcript or a hearing problem. Never say what they meant or didn't mean, like \"you mean X\" or "
    "\"I understand you don't mean Y\"; start straight with the answer about the right term.</note_from_you>"
)

_WRONG = "Your earlier reply, which was wrong: \"{reply}\"\n"
_WRONG_TR = "Yanlış olan önceki cevabın: \"{reply}\"\n"

_NOTE_TR = (
    "<senin_notun>Bunu düzeltme çubuğuna elle yazdın, çünkü az önce verdiğin cevap yanlıştı "
    "veya amacı kaçırdı. Kimse bunu sesli söylemedi. Türkçe veya İngilizce olabilir; her iki şekilde de anla "
    "ve yine Türkçe konuşarak cevap ver.\n"
    "Not: {note}\n{rejected}"
    "Yukarıdaki konuşmayı geçersiz kılar. Konuşma tanımanın bozduğu kelime olabilir, mesela \"rdap\": o zaman "
    "sordukları tam olarak o kelimedir. Yazıldığı gibi al, ona benzeyen veya benzer okunan daha bilinen bir "
    "kelimeyle asla değiştirme. Yanındaki bir ipucu, mesela \"ai model\" veya \"kütüphane\", türünü söyler: o "
    "türde kelimeye en yakın gerçek terimi seç. Böyle bir terimin "
    "var olduğundan ve ne yaptığından emin değilsen adından tahmin ederek asla anlatma; tek cümleyle kullanmadığını söyle ve bildiğin en yakın şeyi anlat; ne yaptığını asla uydurma. "
    "Not bir yönlendirme de olabilir, mesela \"rdap ile ilgili bir örnek ver\" veya \"daha kısa\", o zaman "
    "tam olarak bunu yap. Yanlış bir cevabın konusunu veya yorumunu asla tekrar verme; notu, transkripti "
    "veya duyma sorununu asla okuma, çevirme veya bunlardan bahsetme. \"X'i mi kastettiniz\", "
    "\"Y'yi kastetmediğinizi anlıyorum\" gibi ne kastettiklerini asla söyleme; doğrudan doğru terimle ilgili "
    "cevapla başla.</senin_notun>"
)


# Sent only with a screenshot. The image itself goes right before this text in the same message.
_SCREEN = (
    "<screen>The image above is the screen they are sharing right now, captured by you, not words they said. "
    "Read all of it carefully: every word, number, symbol and line of code. Decide from their speech and the "
    "screen together what they want; if the speech asks for something specific, that wins, and several asks "
    "are answered in the order they asked. A coding task gets the LIVE CODING reply for exactly the task on "
    "the screen; copy the given starter code exactly, with its class, method name, parameters and type hints, "
    "and fill it in. A question or a test item gets its answer, and for multiple choice the option and why. "
    "A matrix, table or calculation: first compare the matrices on the screen cell by cell and note the row "
    "and column of every non-zero number, since one 1 among zeros is the whole point. Then the answer starts "
    "with one short line saying where the non-zero numbers are in each matrix, as row and column, then "
    "the result: the matrix they ask for written out, one row per line like "
    "[0 0 0 1 0], with the numbers exactly right. After it, at most one short sentence of how, like which "
    "side it multiplies from. Never describe the matrix in words or read the numbers one by one. A "
    "diagram, an architecture or code to review gets what it does and the point they ask about, like a bug "
    "or an improvement. Anything else they ask you to write or draw is shown, not described. If a part is too small to read, use what you can read and don't guess the rest. "
    "Never mention a screenshot, an image or a capture; if needed, say \"on the screen\".</screen>"
)
_SCREEN_TR = (
    "<ekran>Yukarıdaki görüntü şu an paylaştıkları ekran, onu sen yakaladın; söyledikleri sözler değil. "
    "Hepsini dikkatle oku: her kelime, sayı, sembol ve kod satırı. Ne istediklerine konuşmaları ve ekran "
    "birlikte bakarak karar ver; konuşmada belirli bir şey istiyorlarsa o geçerlidir, birden fazla şey "
    "sordularsa sordukları sırayla cevapla. Kod görevi, tam olarak ekrandaki görev için CANLI KODLAMA cevabını "
    "alır; ekrandaki başlangıç kodunu sınıfı, metot adı, parametreleri ve tip ipuçlarıyla aynen kopyala ve "
    "içini doldur. Bir soru veya test sorusu cevabını alır, çoktan seçmelide şık ve nedeni. Bir matris, tablo "
    "veya hesap: önce ekrandaki matrisleri hücre hücre karşılaştır ve sıfır olmayan her sayının satır ve "
    "sütununu not et, çünkü sıfırlar arasındaki tek bir 1 bütün meseledir. Sonra cevap girişsiz, doğrudan "
    "sonuçla başlar: önce her matriste sıfır olmayan sayıların yerini satır ve sütun olarak söyleyen tek "
    "kısa bir satır, sonra istedikleri matris yazılır, her satıra bir sıra, mesela [0 0 0 1 0], sayılar tam "
    "doğru. Ardından en fazla bir kısa cümleyle nasıl, mesela hangi taraftan çarpıldığı. Matrisi asla "
    "kelimelerle anlatma, sayıları tek tek okuma. Bir diyagram, mimari veya incelenecek kod ne yaptığını ve "
    "sordukları noktayı alır, mesela bir hata veya iyileştirme. Yazmanı veya çizmeni istedikleri başka bir şey "
    "anlatılmaz, gösterilir. Bir kısım okunamayacak kadar küçükse "
    "okuyabildiğini kullan, gerisini tahmin etme. Ekran görüntüsünden, resimden veya yakalamaktan asla "
    "bahsetme; gerekirse \"ekranda\" de.</ekran>"
)
_NO_SPEECH = "(No new speech since your last reply. They are showing their screen.)"
_NO_SPEECH_TR = "(Son cevabından beri yeni konuşma yok. Ekranlarını gösteriyorlar.)"


@dataclass(frozen=True)
class _Prompt:
    history: Sequence[Turn]   # earlier exchanges, starting with the interviewer and ending with you
    final: str                # the interviewer's newest speech plus the reminder
    image: Shot | None = None # their shared screen, sent before the final text


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
    question: str, history: Sequence[Turn] = (), note: str = "", lang: str = "en", think: bool = False,
    screen: bool = False, rejected: Sequence[str] = (),
) -> str:
    text = question.strip() or (_NO_SPEECH_TR if lang == "tr" else _NO_SPEECH)
    if screen:
        text += "\n\n" + (_SCREEN_TR if lang == "tr" else _SCREEN)
    note_tpl = _NOTE_TR if lang == "tr" else _NOTE
    if lang not in ("en", "tr"):
        note_tpl = note_tpl.replace("spoken English", f"spoken {english_name(lang)}")
    if note := note.strip():
        wrong = "".join(
            (_WRONG_TR if lang == "tr" else _WRONG).format(reply=r) for r in rejected if r.strip()
        )
        text += "\n\n" + note_tpl.format(note=note, rejected=wrong)
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


def claude_messages(history: Sequence[Turn], final: str, image: Shot | None = None) -> list[dict]:
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
    if image is None:
        msgs.append({"role": "user", "content": final})
    else:  # after the cache breakpoint: the image is paid once, at full price, and never cached
        source = {"type": "base64", "media_type": image.media_type, "data": base64.standard_b64encode(image.data).decode()}
        msgs.append({"role": "user", "content": [{"type": "image", "source": source}, {"type": "text", "text": final}]})
    return msgs


def gemini_contents(history: Sequence[Turn], final: str, image: Shot | None = None) -> list[types.Content]:
    contents = [
        types.Content(role="user" if t.role == INTERVIEWER else "model", parts=[types.Part(text=t.text)])
        for t in history
    ]
    parts = [types.Part(text=final)]
    if image is not None:
        parts.insert(0, types.Part.from_bytes(data=image.data, mime_type=image.media_type))
    contents.append(types.Content(role="user", parts=parts))
    return contents


class AnswerEngine:
    """Streams answers from the model picked in the UI; the backups answer only if it fails.

    Chain: the picked model -> the fast Claude backup (Haiku 4.5) -> Gemini; with Gemini picked:
    Gemini -> Sonnet 5 -> Haiku 4.5. The request goes to the picked model alone. Only if it fails
    (bad key, no credit, overload, no first word within the timeout) does the next one get it,
    so a normal answer costs exactly one request.
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
            # every Claude model that can be picked: a key error blocks them all at once
            known = [cfg.claude_model, cfg.claude_backup_model, *MODEL_CHOICES.values()]
            self._claude_models = list(dict.fromkeys(m for m in known if m and m != GEMINI))
        self._blocked_until: dict[str, float] = {}
        self._chain: list[str] = []

        self._gemini: genai.Client | None = None
        self._gemini_model = cfg.gemini_model
        if cfg.gemini_key:
            self._gemini = genai.Client(
                api_key=cfg.gemini_key,
                http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_S * 1000)),
            )
        self._gemini_thinking = 0
        self._gemini_config = self._make_gemini_config()
        self.set_model(cfg.claude_model if self._claude is not None else GEMINI)

    def set_model(self, model: str) -> None:
        """Pick the model that answers; the rest of the chain is only a fallback."""
        backup = self._cfg.claude_backup_model or "claude-haiku-4-5-20251001"
        if model == GEMINI:
            chain = [GEMINI, self._cfg.claude_model or "claude-sonnet-5", backup]
        else:
            chain = [model, backup, GEMINI]
        chain = [m for m in dict.fromkeys(chain) if (self._gemini if m == GEMINI else self._claude) is not None]
        if chain == self._chain:
            return
        self._chain = chain
        self._cache_used_at = 0.0  # the cache belongs to one model: write it for the new one on the next warm()
        log.info("answer chain -> %s", " -> ".join(_label(m) for m in chain))
        self._emit_provider()

    def selected_model(self) -> str:
        return self._chain[0] if self._chain else ""

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
        """The picked model is a Claude model that can think and is usable now (thinking applies only to it)."""
        ready = self._ready_chain()
        return bool(ready) and ready[0] == self.selected_model() != GEMINI and ready[0] not in NO_ADAPTIVE_THINKING

    def _level(self) -> tuple[str | None, int, float, float]:
        return THINKING_LEVELS[self._thinking]

    def _claude_params(self, model: str) -> dict:
        """Thinking settings: only the picked model thinks; the backups always answer at once.

        Each model is switched off its own way (THINKING_OFF_PARAMS); Opus 5.5 can't be, so it runs at low effort.
        """
        effort = self._level()[0] if model == self.selected_model() and model not in NO_ADAPTIVE_THINKING else None
        if effort is None:
            return dict(THINKING_OFF_PARAMS.get(model, {"thinking": {"type": "disabled"}}))
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}

    @staticmethod
    def _thinks(params: dict) -> bool:
        """Thinking happens before the first visible word (Opus 5.5 always thinks: no "thinking" key)."""
        return params.get("thinking", {}).get("type") in ("adaptive", None)

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

    def _ready_chain(self) -> list[str]:
        """The chain without models that are paused after an error (Gemini is never paused)."""
        return [m for m in self._chain if m == GEMINI or self._available(m)]

    def _primary_claude(self) -> str | None:
        """The picked model, if it is a Claude model that can be used right now (for cache and warm-up)."""
        ready = self._ready_chain()
        return ready[0] if ready and ready[0] == self.selected_model() != GEMINI else None

    def active_provider(self) -> str:
        ready = [_label(m) for m in self._ready_chain()]
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
        for model in self._chain:
            if model == GEMINI:
                continue
            try:
                if model == self.selected_model():
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
        primary = self._primary_claude()
        if primary is None:
            return
        now = time.monotonic()
        if now - self._cache_used_at > CACHE_REFRESH_S:
            coro = self._prewarm(primary, history)
        elif now - self._last_activity > WARM_IDLE_S:
            coro = self._claude.models.retrieve(primary)  # type: ignore[union-attr]
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
        image: Shot | None = None,
        rejected: Sequence[str] = (),
    ) -> str:
        """Stream an answer and return the label of the provider that produced it."""
        thinks = self._level()[0] is not None and self.primary_ready()
        final = build_final_message(
            question, history, note, lang=self._lang, think=thinks, screen=image is not None, rejected=rejected
        )
        prompt = _Prompt(tuple(history), final, image)
        self._last_activity = time.monotonic()
        self._emit_provider()  # e.g. back to the primary model after a cooldown
        failures: dict[str, BaseException] = {}
        tried = 0
        for model in self._ready_chain():
            if model != GEMINI and not self._available(model):
                continue  # paused by the previous attempt, e.g. a key error blocks every Claude model
            label = _label(model)
            stream: StreamFn = self._stream_gemini if model == GEMINI else functools.partial(self._stream_claude, model)
            if tried:
                log.info("falling back to %s", label)
            tried += 1
            try:
                await stream(prompt, on_delta)
                return label
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, the next model in the chain takes over
                cause = exc.__cause__ if isinstance(exc, _StreamReset) and exc.__cause__ else exc
                failures[label] = cause
                log.warning("%s failed: %r", label, cause)
                if model != GEMINI:
                    self._classify_claude_error(model, cause)
                if isinstance(exc, _StreamReset):
                    on_reset()  # it broke mid-answer: clear the partial text before the next model writes
        if not tried:
            raise AnswerError("Kullanılabilir model yok (anahtarları kontrol et)")

        last = next(reversed(failures.values()), None)
        raise AnswerError(f"Cevap alınamadı: {_short(last)}")

    async def _stream_claude(self, model: str, prompt: _Prompt, on_delta: Callable[[str], None]) -> None:
        assert self._claude is not None
        if model == self.selected_model():
            self._cache_used_at = time.monotonic()  # cache lifetime counts from request start
        emitted = False
        params = self._claude_params(model)
        if self._thinks(params):  # thinking: more room and time before the first visible word
            level = self._level() if self._level()[0] is not None else THINKING_LEVELS["Low"]  # Opus 5.5 "off"
            _, max_tokens, first_token_s, _ = level
            client = self._claude.with_options(timeout=first_token_s + REQUEST_TIMEOUT_S)
        else:
            max_tokens, first_token_s, client = MAX_TOKENS_CLAUDE, FIRST_TOKEN_TIMEOUT_S, self._claude
        if prompt.image is not None:
            max_tokens = max(max_tokens, SHOT_MAX_TOKENS_CLAUDE)
            first_token_s += SHOT_EXTRA_WAIT_S
        try:
            async with asyncio.timeout(first_token_s) as deadline:
                async with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=self._claude_system,
                    messages=claude_messages(prompt.history, prompt.final, prompt.image),
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
            extra = SHOT_EXTRA_WAIT_S if prompt.image is not None else 0.0
            async with asyncio.timeout(FIRST_TOKEN_TIMEOUT_S + 3 + extra) as deadline:
                stream = await self._gemini.aio.models.generate_content_stream(
                    model=self._gemini_model,
                    contents=gemini_contents(prompt.history, prompt.final, prompt.image),
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
