"""The whole interview in order: interviewer speech segments and the answers already given."""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

_NON_WORD = re.compile(r"[^a-z0-9 ]+")

INTERVIEWER = "interviewer"
YOU = "you"


@dataclass
class Segment:
    text: str
    sent: bool = False


@dataclass(frozen=True)
class Turn:
    role: str                       # INTERVIEWER or YOU
    text: str


@dataclass(frozen=True)
class Candidate:
    key: tuple[int, int, int]       # (pending start, end, answers version); changes with new speech or answers
    question: str                   # everything the interviewer said since your last answer
    history: tuple[Turn, ...]       # the interview before that, oldest first, alternating and starting with them


def _normalize(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def _tail_chars(text: str, max_chars: int) -> str:
    """Keep the end of a very long monologue: the reply is about its last part."""
    if len(text) <= max_chars:
        return text
    cut = text.find(" ", len(text) - max_chars)
    return "…" + text[cut + 1 if cut != -1 else len(text) - max_chars:]


class Conversation:
    """Final transcript segments plus answers; unsent segments always form a suffix of the list."""

    def __init__(self, speech_max_chars: int, history_max_chars: int, min_words: int) -> None:
        self._segments: list[Segment] = []
        self._answers: list[tuple[int, str]] = []   # (segment end the answer replied to, spoken text), sorted
        self._version = 0
        self._speech_max_chars = speech_max_chars
        self._history_max_chars = history_max_chars
        self._min_words = min_words
        self._first_answer = 0                      # older exchanges are dropped once history grows too long
        self._history: tuple[Turn, ...] | None = None

    def add(self, text: str) -> None:
        if text := text.strip():
            self._segments.append(Segment(text))

    def _pending_start(self) -> int:
        start = len(self._segments)
        while start > 0 and not self._segments[start - 1].sent:
            start -= 1
        return start

    def _answered_end(self) -> int:
        return self._answers[-1][0] if self._answers else 0

    def peek(self, allow_empty: bool = False) -> Candidate | None:
        """What Space would reply to now, without consuming it.

        allow_empty: a screenshot is attached, so even no speech or a two-word "solve this" is a request.
        """
        start = self._pending_start()
        end = len(self._segments)
        if start == end and not allow_empty:
            return None
        pending = " ".join(s.text for s in self._segments[start:end])
        if len(_normalize(pending).split()) < self._min_words and not allow_empty:
            return None  # noise or "okay"; stays pending and merges into the next speech
        # Speech committed earlier but never answered (a failed answer) still belongs to this reply.
        begin = min(self._answered_end(), start)
        speech = " ".join(s.text for s in self._segments[begin:end])
        return Candidate(
            key=(start, end, self._version),
            question=_tail_chars(speech, self._speech_max_chars),
            history=self.history(),
        )

    def commit(self, candidate: Candidate) -> None:
        start, end, _ = candidate.key
        for s in self._segments[start:end]:
            s.sent = True

    def record_answer(self, candidate: Candidate, text: str, replace: bool = False, append: bool = False) -> None:
        """Store what was said aloud; a correction replaces the answer to the same speech.

        append: a reply to a screenshot with no new speech follows the previous answer, never replaces it.
        """
        text = text.strip()
        if not text:
            return
        end = candidate.key[1]
        i = bisect.bisect_left(self._answers, end, key=lambda a: a[0])
        if i < len(self._answers) and self._answers[i][0] == end:
            if append and not replace:
                text = f"{self._answers[i][1]} {text}"
            self._answers[i] = (end, text)  # a correction, or a late duplicate for the same speech
        else:
            self._answers.insert(i, (end, text))
        self._version += 1
        self._history = None

    def history(self) -> tuple[Turn, ...]:
        """Alternating interviewer/you turns up to the last answer, oldest first."""
        if self._history is None:
            self._history = self._build_history()
        return self._history

    def _build_history(self) -> tuple[Turn, ...]:
        exchanges: list[tuple[str, str]] = []
        prev = 0
        for end, answer in self._answers:
            speech = " ".join(s.text for s in self._segments[prev:end])
            if exchanges and not speech:
                exchanges[-1] = (exchanges[-1][0], f"{exchanges[-1][1]} {answer}")
            else:
                exchanges.append((speech, answer))
            prev = end
        size = lambda ex: sum(len(q) + len(a) for q, a in ex)  # noqa: E731
        if size(exchanges[self._first_answer:]) > self._history_max_chars:
            # Drop old exchanges in one big step, so the cached prefix changes rarely.
            while self._first_answer < len(exchanges) - 1 and (
                size(exchanges[self._first_answer:]) > self._history_max_chars * 0.7
            ):
                self._first_answer += 1
        turns: list[Turn] = []
        for speech, answer in exchanges[self._first_answer:]:
            turns.append(Turn(INTERVIEWER, speech or "(no speech)"))
            turns.append(Turn(YOU, answer))
        return tuple(turns)
