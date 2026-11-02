"""Screen capture on Hyprland (Wayland): grim grabs the monitor the user is looking at.

Qt cannot read other windows' pixels on Wayland, so the picture comes from grim; Qt only scales
and encodes it. The asyncio thread calls capture(); the slow parts run in a worker thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from PyQt6.QtCore import QBuffer, QIODevice, Qt
from PyQt6.QtGui import QImage

log = logging.getLogger(__name__)

MAX_EDGE = 1568            # the models see at most this many pixels on the long edge; bigger only costs time
MAX_BYTES = 3_500_000      # well under the 5 MB per-image API limit, also after base64
CAPTURE_TIMEOUT_S = 5.0


class CaptureError(Exception):
    pass


@dataclass(frozen=True)
class Shot:
    data: bytes            # encoded image, ready for the API
    media_type: str        # image/png or image/jpeg
    width: int
    height: int
    file: str = ""         # where the session record keeps it (kayitlar/<oturum>_ekran/...)


async def _run(*cmd: str) -> bytes:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise CaptureError(f"{cmd[0]} bulunamadı (sudo pacman -S grim)") from exc
    try:
        out, err = await asyncio.wait_for(proc.communicate(), CAPTURE_TIMEOUT_S)
    except TimeoutError as exc:
        proc.kill()
        raise CaptureError(f"{cmd[0]} zaman aşımı") from exc
    if proc.returncode != 0:
        raise CaptureError(f"{cmd[0]}: {err.decode(errors='replace').strip()[:120]}")
    return out


async def _focused_monitor() -> str | None:
    """The monitor of the active workspace, where the user just switched to (None: let grim pick all)."""
    try:
        return json.loads(await _run("hyprctl", "activeworkspace", "-j")).get("monitor") or None
    except (CaptureError, ValueError) as exc:
        log.warning("focused monitor unknown, capturing all outputs: %s", exc)
        return None


def _encode(png: bytes) -> Shot:
    image = QImage.fromData(png, "PNG")
    if image.isNull():
        raise CaptureError("görüntü okunamadı")
    if max(image.width(), image.height()) > MAX_EDGE:
        image = image.scaled(
            MAX_EDGE, MAX_EDGE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
    for fmt, media_type, quality in (("PNG", "image/png", -1), ("JPEG", "image/jpeg", 90)):
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buf, fmt, quality)
        data = bytes(buf.data())
        if len(data) <= MAX_BYTES:  # PNG keeps code and small text sharp; JPEG only for busy photos
            return Shot(data, media_type, image.width(), image.height())
    raise CaptureError("görüntü çok büyük")


async def capture() -> Shot:
    monitor = await _focused_monitor()
    png = await _run("grim", *(("-o", monitor) if monitor else ()), "-t", "png", "-l", "1", "-")
    shot = await asyncio.to_thread(_encode, png)
    log.info("screenshot %dx%d %s %d KB (monitor %s)", shot.width, shot.height, shot.media_type,
             len(shot.data) // 1024, monitor or "all")
    return shot
