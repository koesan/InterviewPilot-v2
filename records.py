"""Session records: the interviewer transcript and the answers, each in its own text file."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TextIO

log = logging.getLogger(__name__)

QUESTION_CHARS = 300  # the end of the speech an answer replied to, so the answers file reads on its own


class SessionRecord:
    """Appends line by line while the interview runs, so nothing is lost if the app closes or crashes."""

    def __init__(self, folder: Path) -> None:
        stamp = time.strftime("%Y-%m-%d_%H-%M")
        self.transcript_path = folder / f"{stamp}_transkript.txt"
        self.answers_path = folder / f"{stamp}_cevaplar.txt"
        self._transcript = self._open(self.transcript_path)
        self._answers = self._open(self.answers_path)

    @staticmethod
    def _open(path: Path) -> TextIO | None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:  # a full disk must never stop the interview
            log.error("kayıt dosyası açılamadı %s: %s", path, exc)
            return None

    @staticmethod
    def _write(f: TextIO | None, text: str) -> None:
        if f is None:
            return
        try:
            f.write(text)
        except OSError as exc:
            log.error("kayıt yazılamadı: %s", exc)

    def speech(self, text: str) -> None:
        self._write(self._transcript, f"[{time.strftime('%H:%M:%S')}] {text}\n")

    def answer(self, number: int, question: str, answer: str, provider: str, note: str = "") -> None:
        stamp = time.strftime("%H:%M:%S")
        head = f"#{number}  {stamp}  {provider}" + (f"  · düzeltme: {note}" if note else "")
        if len(question) > QUESTION_CHARS:
            question = "…" + question[-(QUESTION_CHARS - 1):]
        self._write(self._answers, f"{head}\nSoru: {question}\nCevap: {answer}\n\n")
        # The answer in the transcript too, where Space was pressed, so the interview reads as one dialogue.
        fix = f" · düzeltme: {note}" if note else ""
        self._write(self._transcript, f"[{stamp}] >>> CEVAP #{number} ({provider}{fix}): {answer}\n")

    def close(self) -> None:
        for f in (self._transcript, self._answers):
            if f is not None:
                f.close()
