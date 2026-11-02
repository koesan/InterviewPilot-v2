"""Offline checks: no API key, no audio, no network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROMPTS_DIR, _read_text  # noqa: E402
from languages import LANGUAGES, TRANSLATION_TARGET, english_name  # noqa: E402
from llm import _language_rule, build_final_message  # noqa: E402
from translate import _system  # noqa: E402


def test_ten_languages():
    assert len(LANGUAGES) == 10
    assert {"en", "tr", "zh", "hi", "es", "fr", "ar", "bn", "pt", "ru"} == set(LANGUAGES)
    assert TRANSLATION_TARGET in LANGUAGES


def test_example_prompts_exist():
    for name in ("prompt", "cv", "job", "prompt_tr", "cv_tr", "job_tr"):
        assert (PROMPTS_DIR / f"{name}.example.txt").read_text(encoding="utf-8").strip()


def test_falls_back_to_example(tmp_path):
    (tmp_path / "cv.example.txt").write_text("example cv", encoding="utf-8")
    assert _read_text(tmp_path / "cv.txt") == "example cv"
    (tmp_path / "cv.txt").write_text("own cv", encoding="utf-8")
    assert _read_text(tmp_path / "cv.txt") == "own cv"


def test_reminder_per_language():
    assert "Reply only in Spanish" in build_final_message("Hola", lang="es")
    assert "<reminder>" in build_final_message("Hello", lang="en")
    assert "Reply only" not in build_final_message("Hello", lang="en")
    assert "<hatırlatma>" in build_final_message("Merhaba", lang="tr")
    note = build_final_message("Bonjour", note="plus court", lang="fr")
    assert "spoken French" in note and "spoken English" not in note


def test_language_rule_and_translation_prompt():
    for code in LANGUAGES:
        name = english_name(code)
        assert name in _language_rule(code)
        assert f"live {name} speech" in _system(name, ("OpenCV",))
