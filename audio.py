"""System-audio capture (PipeWire/PulseAudio monitor) via parec."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from config import AUDIO_LATENCY_MS, AUDIO_READ_BYTES, SAMPLE_RATE

log = logging.getLogger(__name__)


class AudioCapture:
    """Streams 16 kHz mono s16le PCM to `on_chunk`; restarts parec if it dies."""

    def __init__(
        self,
        source: str,
        on_chunk: Callable[[bytes], Awaitable[None]],
        on_error: Callable[[str], None],
    ) -> None:
        self._source = source
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._proc: asyncio.subprocess.Process | None = None

    async def run(self) -> None:
        backoff = 0.5
        while True:
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    "parec",
                    f"--device={self._source}",
                    "--raw",
                    "--format=s16le",
                    f"--rate={SAMPLE_RATE}",
                    "--channels=1",
                    f"--latency-msec={AUDIO_LATENCY_MS}",
                    "--client-name=interview-assistant",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                self._on_error("parec bulunamadı — 'sudo pacman -S libpulse' kur")
                return

            log.info("parec started (source=%s)", self._source)
            try:
                assert self._proc.stdout is not None
                while chunk := await self._proc.stdout.read(AUDIO_READ_BYTES):
                    backoff = 0.5
                    await self._on_chunk(chunk)
            finally:
                await self._terminate()

            err = b""
            if self._proc.stderr is not None:
                err = await self._proc.stderr.read()
            msg = err.decode(errors="replace").strip() or f"çıkış kodu {self._proc.returncode}"
            log.warning("parec stopped: %s", msg)
            self._on_error(f"Ses yakalama durdu, yeniden başlatılıyor: {msg[:80]}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)

    async def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 1.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()

    async def close(self) -> None:
        await self._terminate()
