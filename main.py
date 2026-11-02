"""Entry point: Qt on the main thread, all network/audio work on one asyncio thread."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import sys
import threading
from concurrent.futures import Future
from logging.handlers import RotatingFileHandler
from typing import Any, Coroutine

from PyQt6.QtCore import QSocketNotifier
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

from config import LOG_FILE, PID_FILE, STYLE_FILE, load_config
from core import Core
from ui import Bridge, MainWindow

log = logging.getLogger("assistant")

FONT_CANDIDATES = ("Inter", "Noto Sans", "Cantarell", "DejaVu Sans")


class AsyncRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="asyncio", daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> None:
        self._thread.start()

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def call(self, fn: Any, *args: Any) -> None:
        self.loop.call_soon_threadsafe(fn, *args)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2)


def setup_logging() -> None:
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console])
    for noisy in ("httpx", "httpx2", "httpcore", "websockets", "google_genai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def install_unix_signals(app: QApplication, on_toggle: Any, on_shot: Any) -> QSocketNotifier:
    """SIGUSR1 toggles (toggle.sh), SIGUSR2 takes a screenshot (shot.sh); SIGINT/SIGTERM quit cleanly."""
    rsock, wsock = socket.socketpair()
    rsock.setblocking(False)
    wsock.setblocking(False)
    signal.set_wakeup_fd(wsock.fileno(), warn_on_full_buffer=False)
    for sig in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: None)  # real handling happens in the notifier below

    def dispatch() -> None:
        try:
            data = rsock.recv(64)
        except BlockingIOError:
            return
        for signum in data:
            if signum == signal.SIGUSR1:
                on_toggle()
            elif signum == signal.SIGUSR2:
                on_shot()
            elif signum in (signal.SIGINT, signal.SIGTERM):
                app.quit()

    notifier = QSocketNotifier(rsock.fileno(), QSocketNotifier.Type.Read, app)
    notifier.activated.connect(dispatch)
    notifier._socks = (rsock, wsock)  # keep sockets alive with the notifier
    return notifier


def pick_font() -> QFont:
    families = set(QFontDatabase.families())
    family = next((f for f in FONT_CANDIDATES if f in families), "sans-serif")
    font = QFont(family)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


def main() -> int:
    setup_logging()
    cfg = load_config()
    log.info("starting (claude=%s, gemini=%s, source=%s)", cfg.claude_model, cfg.gemini_model, cfg.audio_source)

    app = QApplication(sys.argv)
    app.setApplicationName("Interview Assistant")
    app.setFont(pick_font())
    try:
        app.setStyleSheet(STYLE_FILE.read_text(encoding="utf-8"))
    except OSError as exc:
        log.error("style.qss not loaded: %s", exc)

    bridge = Bridge()
    window = MainWindow(bridge, cfg.claude_model if cfg.anthropic_key else "gemini")

    runner = AsyncRunner()
    runner.start()

    async def build() -> Core:
        return Core(cfg, bridge)

    core: Core = runner.submit(build()).result(timeout=10)
    window.toggle_requested.connect(lambda: runner.call(core.toggle))
    window.retry_requested.connect(lambda hint: runner.call(core.retry, hint))
    window.language_changed.connect(lambda lang: runner.submit(core.set_language(lang)))
    window.thinking_changed.connect(lambda label: runner.call(core.set_thinking, label))
    window.model_changed.connect(lambda label: runner.call(core.set_model, label))
    window.translation_toggled.connect(lambda on: runner.call(core.set_translation, on))
    window.screenshot_requested.connect(lambda: runner.call(core.screenshot))
    window.screenshot_cleared.connect(lambda: runner.call(core.clear_screenshot))
    notifier = install_unix_signals(app, lambda: runner.call(core.toggle), lambda: runner.call(core.screenshot))

    try:
        PID_FILE.write_text(str(os.getpid()))
    except OSError as exc:
        log.warning("pid file not written: %s", exc)

    window.show()
    runner.submit(core.run())
    code = app.exec()

    try:
        runner.submit(core.shutdown()).result(timeout=3)
    except Exception as exc:  # noqa: BLE001 - never block exit
        log.warning("shutdown incomplete: %r", exc)
    runner.stop()
    notifier.setEnabled(False)
    try:
        if PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()
    except OSError:
        pass
    log.info("exited")
    return code


if __name__ == "__main__":
    sys.exit(main())
