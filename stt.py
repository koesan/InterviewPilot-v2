"""Deepgram Nova-3 live transcription over a raw WebSocket."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from urllib.parse import urlencode

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, InvalidStatus

from config import (
    ABSORB_LATE_FINAL_S,
    DG_ENDPOINTING_MS,
    DG_KEEPALIVE_AFTER_S,
    DG_UTTERANCE_END_MS,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)

DG_URL = "wss://api.deepgram.com/v1/listen"
_FINALIZE = json.dumps({"type": "Finalize"})
_KEEPALIVE = json.dumps({"type": "KeepAlive"})
_CLOSE = json.dumps({"type": "CloseStream"})


class DeepgramSTT:
    """Keeps one streaming connection alive, reconnecting with backoff.

    Callbacks run on the asyncio loop:
      on_interim(text)            – provisional text for the current utterance
      on_final(text, end_of_turn) – locked-in text (may be empty); end_of_turn on speech_final
      on_utterance_end()          – ~1 s of silence after speech
      on_status(text, ok)         – connection state for the UI
    """

    def __init__(
        self,
        api_key: str,
        keyterms: tuple[str, ...],
        on_interim: Callable[[str], None],
        on_final: Callable[[str, bool], None],
        on_utterance_end: Callable[[], None],
        on_status: Callable[[str, bool], None],
        language: str = "en",
        model: str = "nova-3",
    ) -> None:
        self._headers = {"Authorization": f"Token {api_key}"}
        self._keyterms = keyterms
        self._language = language
        self._model = model
        self._url = self._build_url()
        self._on_interim = on_interim
        self._on_final = on_final
        self._on_utterance_end = on_utterance_end
        self._on_status = on_status
        self._ws: ClientConnection | None = None
        self._last_send = 0.0
        self._pending_interim = False
        self._interim_text = ""
        self._absorb_until = 0.0
        self._finalized = asyncio.Event()
        self._reconnect_requested = False

    def _build_url(self) -> str:
        params = [
            ("model", self._model),
            ("language", self._language),
            ("encoding", "linear16"),
            ("sample_rate", str(SAMPLE_RATE)),
            ("channels", "1"),
            ("interim_results", "true"),
            ("smart_format", "true"),
            ("punctuate", "true"),
            ("endpointing", str(DG_ENDPOINTING_MS)),
            ("utterance_end_ms", str(DG_UTTERANCE_END_MS)),
            ("vad_events", "true"),
        ]
        params += [("keyterm", k) for k in self._keyterms]
        return f"{DG_URL}?{urlencode(params)}"

    async def reconnect_with_language(self, language: str) -> None:
        """Switch STT language by closing the current connection and rebuilding the URL."""
        if language == self._language:
            return
        self._language = language
        self._url = self._build_url()
        log.info("STT language changed to %s, reconnecting", language)
        self._reconnect_requested = True
        ws = self._ws
        if ws is not None:
            # Force-close so run()'s "async for message in ws" exits immediately
            try:
                await asyncio.wait_for(ws.close(), 0.5)
            except (ConnectionClosed, OSError, TimeoutError):
                pass

    # ---- connection lifecycle -------------------------------------------------

    async def run(self) -> None:
        backoff = 0.5
        while True:
            self._on_status("Deepgram: bağlanıyor…", False)
            try:
                async with websockets.connect(
                    self._url,
                    additional_headers=self._headers,
                    open_timeout=6,
                    ping_interval=10,
                    ping_timeout=10,
                    max_size=None,
                ) as ws:
                    self._ws = ws
                    self._last_send = time.monotonic()
                    self._pending_interim = False
                    self._reconnect_requested = False
                    backoff = 0.5
                    self._on_status("Deepgram: bağlı", True)
                    log.info("deepgram connected (language=%s)", self._language)
                    keepalive = asyncio.create_task(self._keepalive(ws))
                    try:
                        async for message in ws:
                            if isinstance(message, str):
                                self._handle(message)
                    finally:
                        keepalive.cancel()
                        self._ws = None
                log.info("deepgram closed: %s", ws.close_code)
            except InvalidStatus as exc:
                code = exc.response.status_code
                log.error("deepgram handshake rejected: %s", code)
                if code in (401, 403):
                    self._on_status("Deepgram: API anahtarı geçersiz", False)
                    return
                self._on_status(f"Deepgram: HTTP {code}, tekrar deneniyor", False)
            except (OSError, ConnectionClosed, TimeoutError) as exc:
                log.warning("deepgram connection error: %r", exc)
                self._on_status("Deepgram: bağlantı koptu, tekrar deneniyor", False)
            self._ws = None
            self._clear_interim()
            # Language switch: reconnect immediately without backoff
            if self._reconnect_requested:
                self._reconnect_requested = False
                log.info("reconnecting immediately for language switch")
                continue
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

    async def _keepalive(self, ws: ClientConnection) -> None:
        while True:
            await asyncio.sleep(1.0)
            if time.monotonic() - self._last_send >= DG_KEEPALIVE_AFTER_S:
                await self._send(ws, _KEEPALIVE)

    async def close(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(_CLOSE)
            await asyncio.wait_for(ws.close(), 1.0)
        except (ConnectionClosed, OSError, TimeoutError):
            pass

    # ---- sending ------------------------------------------------------------------

    async def _send(self, ws: ClientConnection, data: bytes | str) -> bool:
        try:
            await ws.send(data)
        except (ConnectionClosed, OSError):
            return False  # the receive loop notices and reconnects
        self._last_send = time.monotonic()
        return True

    async def send_audio(self, chunk: bytes) -> None:
        ws = self._ws
        if ws is not None:
            await self._send(ws, chunk)

    async def finalize(self, max_wait: float) -> str:
        """Flush buffered audio after a pause.

        Returns "" when the flushed text arrived through on_final (or nothing was pending).
        If the flush is slower than `max_wait`, returns the current interim text instead so
        the caller can answer right away; the late flush result is then dropped.
        """
        ws = self._ws
        if ws is None:
            return self._take_interim()
        self._finalized.clear()
        must_wait = self._pending_interim
        if not await self._send(ws, _FINALIZE):
            return self._take_interim()
        if not must_wait:
            return ""
        try:
            await asyncio.wait_for(self._finalized.wait(), max_wait)
            return ""
        except TimeoutError:
            log.info("finalize slower than %.2fs, using interim text", max_wait)
            self._absorb_until = time.monotonic() + ABSORB_LATE_FINAL_S
            return self._take_interim()

    # ---- receiving ----------------------------------------------------------------

    def _take_interim(self) -> str:
        text = self._interim_text if self._pending_interim else ""
        self._pending_interim = False
        self._interim_text = ""
        return text

    def _clear_interim(self) -> None:
        if self._pending_interim:
            self._take_interim()
            self._on_interim("")

    def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        kind = msg.get("type")
        if kind == "UtteranceEnd":
            self._on_utterance_end()
            return
        if kind != "Results":
            return
        try:
            text = msg["channel"]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError, TypeError):
            text = ""

        if self._absorb_until:
            # Pre-pause audio whose interim text was already used as the question.
            if time.monotonic() < self._absorb_until:
                if msg.get("from_finalize"):
                    self._absorb_until = 0.0
                return
            self._absorb_until = 0.0

        if msg.get("is_final"):
            self._pending_interim = False
            self._interim_text = ""
            self._on_final(text, bool(msg.get("speech_final")))
            if msg.get("from_finalize"):
                self._finalized.set()
        elif text:
            self._pending_interim = True
            self._interim_text = text
            self._on_interim(text)
