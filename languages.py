"""Interview languages: Deepgram language code -> (name shown in the UI, English name for the prompts).

English and Turkish have their own prompt files; every other language reuses the English prompt with a
language rule added, so the same answer style works in any of them.
"""
from __future__ import annotations

LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("English", "English"),
    "tr": ("Türkçe", "Turkish"),
    "zh": ("中文", "Mandarin Chinese"),
    "hi": ("हिन्दी", "Hindi"),
    "es": ("Español", "Spanish"),
    "fr": ("Français", "French"),
    "ar": ("العربية", "Arabic"),
    "bn": ("বাংলা", "Bengali"),
    "pt": ("Português", "Portuguese"),
    "ru": ("Русский", "Russian"),
}
DEFAULT_LANGUAGE = "en"
TRANSLATION_TARGET = "tr"   # the transcript is translated into this language for the candidate


def display_name(code: str) -> str:
    return LANGUAGES.get(code, (code, code))[0]


def english_name(code: str) -> str:
    return LANGUAGES.get(code, (code, code))[1]
