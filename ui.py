"""PyQt6 interface: status bar, live transcript (English on top, Turkish translation below), streaming
answer cards."""
from __future__ import annotations

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut, QTextBlockFormat, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import THINKING_LEVELS, THINKING_OFF
from languages import DEFAULT_LANGUAGE, LANGUAGES, TRANSLATION_TARGET

NOTICE_MS = 10_000
FIX_HELP = "Ctrl+E ile yaz  ·  Enter gönderir  ·  son soruyu bu nota göre yeniden cevaplar"
READY_HELP = "Yazmaya başla, Enter gönderir  ·  boşken Space dinlemeye devam eder"
TYPING_HELP = "⌨  Space artık boşluk yazar  ·  Enter gönderir, Esc çıkar"


class Bridge(QObject):
    """Signals emitted from the asyncio thread; Qt queues them onto the GUI thread."""

    listening_changed = pyqtSignal(bool)
    audio_active = pyqtSignal(bool)
    stt_status = pyqtSignal(str, bool)
    provider = pyqtSignal(str)
    notice = pyqtSignal(str)
    interim = pyqtSignal(str)
    final = pyqtSignal(str, bool)
    paragraph_break = pyqtSignal()
    answer_started = pyqtSignal(int, str)
    answer_delta = pyqtSignal(int, str)
    answer_reset = pyqtSignal(int)
    answer_done = pyqtSignal(int, str)
    answer_failed = pyqtSignal(int, str)
    translation_started = pyqtSignal()
    translation_delta = pyqtSignal(str)


def _repolish(widget: QWidget) -> None:
    for w in (widget, *widget.findChildren(QWidget)):
        w.style().unpolish(w)
        w.style().polish(w)


class StickyScroll:
    """Follow new content only while the user is at the bottom."""

    def __init__(self, bar: QScrollBar, threshold: int = 48) -> None:
        self._bar = bar
        self._threshold = threshold
        self._stick = True
        bar.valueChanged.connect(self._on_value)
        bar.rangeChanged.connect(self._on_range)

    def _on_value(self, value: int) -> None:
        self._stick = value >= self._bar.maximum() - self._threshold

    def _on_range(self, _lo: int, hi: int) -> None:
        if self._stick:
            self._bar.setValue(hi)


class TranscriptView(QTextEdit):
    """Final text in the main colour, the live interim tail dimmed and replaced in place."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("transcript")
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.document().setDocumentMargin(18)

        self._final_fmt = QTextCharFormat()
        self._final_fmt.setForeground(QColor("#e7eaf0"))
        self._interim_fmt = QTextCharFormat()
        self._interim_fmt.setForeground(QColor("#6f7888"))
        self._block_fmt = QTextBlockFormat()
        self._block_fmt.setLineHeight(145, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        self._block_fmt.setBottomMargin(14)
        QTextCursor(self.document()).setBlockFormat(self._block_fmt)

        self._interim_start = 0
        self._paragraph_open = False
        self._sticky = StickyScroll(self.verticalScrollBar())

    def _cursor_at_interim(self) -> QTextCursor:
        cur = QTextCursor(self.document())
        cur.setPosition(self._interim_start)
        cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cur.removeSelectedText()
        return cur

    def set_interim(self, text: str) -> None:
        cur = self._cursor_at_interim()
        if text:
            cur.insertText((" " if self._paragraph_open else "") + text, self._interim_fmt)

    def add_final(self, text: str, end_of_turn: bool) -> None:
        cur = self._cursor_at_interim()
        if text:
            cur.insertText((" " if self._paragraph_open else "") + text, self._final_fmt)
            self._paragraph_open = True
        self._interim_start = cur.position()
        if end_of_turn:
            self.break_paragraph()

    def break_paragraph(self) -> None:
        cur = self._cursor_at_interim()
        if self._paragraph_open:
            cur.insertBlock(self._block_fmt, self._final_fmt)
            self._paragraph_open = False
        self._interim_start = cur.position()


class TranslationView(QTextEdit):
    """Turkish translation, one paragraph per piece of interviewer speech, streamed in."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("translation")
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.document().setDocumentMargin(18)
        self._fmt = QTextCharFormat()
        self._fmt.setForeground(QColor("#d6dbe4"))
        self._block_fmt = QTextBlockFormat()
        self._block_fmt.setLineHeight(140, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        self._block_fmt.setBottomMargin(12)
        QTextCursor(self.document()).setBlockFormat(self._block_fmt)
        self._empty = True
        self._sticky = StickyScroll(self.verticalScrollBar())

    def start(self) -> None:
        cur = QTextCursor(self.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        if not self._empty:
            cur.insertBlock(self._block_fmt, self._fmt)
        self._empty = False

    def reset(self) -> None:
        self.clear()
        QTextCursor(self.document()).setBlockFormat(self._block_fmt)
        self._empty = True

    def delta(self, text: str) -> None:
        cur = QTextCursor(self.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertText(text.lstrip() if self._at_block_start(cur) else text, self._fmt)

    @staticmethod
    def _at_block_start(cur: QTextCursor) -> bool:
        return cur.positionInBlock() == 0


class AnswerCard(QFrame):
    def __init__(self, question: str) -> None:
        super().__init__()
        self.setObjectName("card")
        self._text = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 16, 22, 18)
        layout.setSpacing(8)

        q = question if len(question) <= 160 else "…" + question[-159:]
        self._question = self._label("cardQuestion", q)
        self._answer = self._label("cardAnswer", "…")
        self._meta = self._label("cardMeta", "")
        self._meta.hide()
        for w in (self._question, self._answer, self._meta):
            layout.addWidget(w)
        self.set_latest(True)

    @staticmethod
    def _label(name: str, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName(name)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return label

    def set_latest(self, latest: bool) -> None:
        self.setProperty("latest", latest)
        _repolish(self)

    def append(self, text: str) -> None:
        self._text += text
        self._answer.setText(self._text.lstrip())

    def reset(self) -> None:
        self._text = ""
        self._answer.setText("…")

    def finish(self, meta: str, error: bool = False) -> None:
        if error:
            self._answer.setProperty("error", True)
            self._answer.setText(meta)
            _repolish(self._answer)
            return
        self._answer.setText(self._text.strip())
        self._meta.setText(meta)
        self._meta.show()


class AnswersView(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("answers")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setObjectName("answersContainer")
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(18, 18, 18, 18)
        self._layout.setSpacing(14)
        self._hint = QLabel("Görüşmeci sorusunu bitirince Space'e bas.\nCevap burada belirecek.")
        self._hint.setObjectName("hint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._hint)
        self._layout.addStretch(1)
        self.setWidget(container)

        self._cards: dict[int, AnswerCard] = {}
        self._latest: AnswerCard | None = None
        self._sticky = StickyScroll(self.verticalScrollBar())

    def start(self, answer_id: int, question: str) -> None:
        self._hint.hide()
        if self._latest is not None:
            self._latest.set_latest(False)
        card = AnswerCard(question)
        self._cards[answer_id] = card
        self._latest = card
        self._layout.insertWidget(self._layout.count() - 1, card)

    def delta(self, answer_id: int, text: str) -> None:
        if card := self._cards.get(answer_id):
            card.append(text)

    def reset(self, answer_id: int) -> None:
        if card := self._cards.get(answer_id):
            card.reset()

    def done(self, answer_id: int, meta: str) -> None:
        if card := self._cards.get(answer_id):
            card.finish(meta)

    def failed(self, answer_id: int, message: str) -> None:
        if card := self._cards.get(answer_id):
            card.finish(message, error=True)


def _panel(title: str, body: QWidget) -> QWidget:
    """A titled column; the title label is reachable as panel.title."""
    panel = QWidget()
    panel.setObjectName("panel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    header = QLabel(title)
    header.setObjectName("panelTitle")
    layout.addWidget(header)
    layout.addWidget(body, 1)
    panel.title = header  # type: ignore[attr-defined]
    return panel


class TermEdit(QLineEdit):
    """Free-text note for the model: a garbled term, or what the answer should do.

    The box is focused automatically on pause so a note can be typed without reaching for
    the mouse, which would normally cost the Space control. It does not: while the box is
    EMPTY, Space still toggles listening - a leading space has nothing to join anyway. Once
    there is text, Space types a space like any text field and the bar says so; Enter or
    Esc then hands focus back.
    """

    space_pressed = pyqtSignal()
    focus_changed = pyqtSignal(bool)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space and not event.modifiers() and not self.text():
            if not event.isAutoRepeat():
                self.space_pressed.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.clearFocus()  # back out without sending; Space toggles again
            event.accept()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        self.focus_changed.emit(True)

    def focusOutEvent(self, event) -> None:  # type: ignore[override]
        super().focusOutEvent(event)
        self.focus_changed.emit(False)


class MainWindow(QMainWindow):
    toggle_requested = pyqtSignal()
    retry_requested = pyqtSignal(str)   # a term typed on the correction bar
    language_changed = pyqtSignal(str)  # a key of LANGUAGES, e.g. "en"
    thinking_changed = pyqtSignal(str)  # a key of THINKING_LEVELS
    translation_toggled = pyqtSignal(bool)  # translation pane shown or hidden

    def __init__(self, bridge: Bridge) -> None:
        super().__init__()
        self.setWindowTitle("Interview Assistant")
        self.resize(1440, 880)

        self._status = QLabel()
        self._status.setObjectName("status")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        key_hint = QLabel("Space  ·  dinlemeyi durdur / devam et")
        key_hint.setObjectName("keyHint")
        self._notice = QLabel()
        self._notice.setObjectName("notice")
        self._notice.setMaximumWidth(460)  # long errors must not widen the window
        self._sound = QLabel()
        self._sound.setObjectName("chip")
        self._stt = QLabel("Deepgram: —")
        self._stt.setObjectName("chip")
        self._llm = QLabel("LLM: —")
        self._llm.setObjectName("chip")

        self._lang_combo = QComboBox()
        self._lang_combo.setObjectName("langCombo")
        for code, (name, _) in LANGUAGES.items():
            self._lang_combo.addItem(name, code)
        self._lang_combo.setCurrentIndex(self._lang_combo.findData(DEFAULT_LANGUAGE))
        self._lang_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lang_combo.setToolTip("Mülakat dili: konuşma tanıma ve cevap bu dilde")
        self._lang_combo.currentIndexChanged.connect(self._on_language)

        think_label = QLabel("Düşünme")
        think_label.setObjectName("keyHint")
        self._think_combo = QComboBox()
        self._think_combo.setObjectName("langCombo")
        self._think_combo.addItems(list(THINKING_LEVELS))
        self._think_combo.setCurrentText(THINKING_OFF)
        self._think_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._think_combo.setToolTip("Kapalı: en hızlı. Low/Medium/High: önce düşünür, daha geç ama daha özenli cevap.")
        self._think_combo.currentTextChanged.connect(self.thinking_changed.emit)

        # Which transcript panes are visible: what they said and its Turkish translation
        self._show_en = self._toggle_button("EN", "Transkripti göster / gizle")
        self._show_tr = self._toggle_button("TR", "Türkçe çeviriyi göster / gizle")

        top = QWidget()
        top.setObjectName("topBar")
        bar = QHBoxLayout(top)
        bar.setContentsMargins(20, 14, 20, 14)
        bar.setSpacing(16)
        bar.addWidget(self._lang_combo)
        bar.addWidget(think_label)
        bar.addWidget(self._think_combo)
        bar.addWidget(self._show_en)
        bar.addWidget(self._show_tr)
        bar.addWidget(self._status)
        bar.addWidget(key_hint)
        bar.addStretch(1)
        bar.addWidget(self._notice)
        bar.addWidget(self._sound)
        bar.addWidget(self._stt)
        bar.addWidget(self._llm)

        self.term = TermEdit()
        self.term.setObjectName("term")
        self.term.setPlaceholderText("rdap   ·   rdap ile ilgili bir örnek ver   ·   daha kısa cevapla")
        self.term.setMaxLength(200)
        self.term.setMinimumWidth(520)
        # ClickFocus keeps it out of the tab chain, so Qt never hands it focus on its own:
        # the box is only entered by clicking it or with Ctrl+E, and Space toggles until then.
        self.term.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.term.space_pressed.connect(self.toggle_requested.emit)
        self.term.focus_changed.connect(self._sync_help)
        self.term.returnPressed.connect(self._emit_retry)
        self.term.textChanged.connect(lambda _: (self._sync_fix(), self._sync_help()))
        self._fix = QPushButton("Yeniden cevapla")
        self._fix.setObjectName("fixBtn")
        self._fix.setEnabled(False)
        self._fix.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._fix.clicked.connect(self._emit_retry)
        fix_label = QLabel("Not")
        fix_label.setObjectName("fixLabel")
        self._fix_help = QLabel(FIX_HELP)
        self._fix_help.setObjectName("fixHelp")

        fix_bar = QWidget()
        fix_bar.setObjectName("fixBar")
        fixes = QHBoxLayout(fix_bar)
        fixes.setContentsMargins(20, 10, 20, 10)
        fixes.setSpacing(12)
        fixes.addWidget(fix_label)
        fixes.addWidget(self.term)
        fixes.addWidget(self._fix)
        fixes.addWidget(self._fix_help)
        fixes.addStretch(1)

        self.transcript = TranscriptView()
        self.translation = TranslationView()
        self.answers = AnswersView()
        self._en_panel = _panel("TRANSCRIPT · EN", self.transcript)
        self._tr_panel = _panel("ÇEVİRİ · TR", self.translation)
        left = QSplitter(Qt.Orientation.Vertical)
        left.setChildrenCollapsible(False)
        left.setHandleWidth(1)
        left.addWidget(self._en_panel)
        left.addWidget(self._tr_panel)
        left.setSizes([440, 440])
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        splitter.addWidget(left)
        splitter.addWidget(_panel("ANSWERS", self.answers))
        splitter.setSizes([700, 740])
        self._show_en.toggled.connect(self._sync_panes)
        self._show_tr.toggled.connect(self._sync_panes)
        self._show_tr.toggled.connect(self.translation_toggled.emit)

        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top)
        layout.addWidget(fix_bar)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        for w in (self, root, top, fix_bar, splitter, left):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Nothing holds focus at startup, so Space toggles from the moment the app opens.
        # Ctrl+E puts the cursor in the note box; Enter or Esc gives focus back.
        focus_note = QShortcut(QKeySequence("Ctrl+E"), self)
        focus_note.setContext(Qt.ShortcutContext.ApplicationShortcut)
        focus_note.activated.connect(self.term.setFocus)

        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut.setAutoRepeat(False)
        shortcut.activated.connect(self.toggle_requested.emit)

        self._notice_timer = QTimer(self, singleShot=True, interval=NOTICE_MS)
        self._notice_timer.timeout.connect(self._notice.clear)

        bridge.listening_changed.connect(self._set_listening)
        bridge.audio_active.connect(self._set_sound)
        bridge.stt_status.connect(self._set_stt)
        bridge.provider.connect(lambda name: self._llm.setText(f"LLM: {name}"))
        bridge.notice.connect(self.show_notice)
        bridge.interim.connect(self.transcript.set_interim)
        bridge.final.connect(self.transcript.add_final)
        bridge.paragraph_break.connect(self.transcript.break_paragraph)
        bridge.answer_started.connect(self.answers.start)
        bridge.answer_delta.connect(self.answers.delta)
        bridge.answer_reset.connect(self.answers.reset)
        bridge.answer_done.connect(self.answers.done)
        bridge.answer_failed.connect(self.answers.failed)
        bridge.translation_started.connect(self.translation.start)
        bridge.translation_delta.connect(self.translation.delta)
        self._listening = True
        self._set_listening(True)
        self._set_sound(False)

    @staticmethod
    def _toggle_button(text: str, tip: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("paneBtn")
        button.setCheckable(True)
        button.setChecked(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space must keep toggling listening
        button.setToolTip(tip)
        return button

    def _current_language(self) -> str:
        return self._lang_combo.currentData() or DEFAULT_LANGUAGE

    def _on_language(self, *_: object) -> None:
        code = self._current_language()
        turkish = code == TRANSLATION_TARGET
        # Turkish speech needs no translation: one Turkish transcript, the TR pane is not used.
        self._en_panel.title.setText(f"TRANSCRIPT · {code.upper()}")
        self._show_en.setText("Metin" if turkish else code.upper())
        self._show_tr.setEnabled(not turkish)
        self.translation.reset()
        self._sync_panes()
        self.language_changed.emit(code)

    def _sync_panes(self, *_: object) -> None:
        """Show the chosen panes; at least one transcript pane always stays visible."""
        turkish = self._current_language() == TRANSLATION_TARGET
        en, tr = self._show_en.isChecked(), self._show_tr.isChecked() and not turkish
        if not en and not tr:
            sender = self.sender()
            other = self._show_tr if sender is self._show_en and not turkish else self._show_en
            other.setChecked(True)  # re-enters this method with a valid state
            return
        self._en_panel.setVisible(en)
        self._tr_panel.setVisible(tr)

    def _set_sound(self, active: bool) -> None:
        self._sound.setText("Ses: geliyor" if active else "Ses: sessiz")
        self._sound.setProperty("level", "on" if active else "off")
        _repolish(self._sound)

    def _set_listening(self, listening: bool) -> None:
        self._listening = listening
        self._status.setText("●  LISTENING" if listening else "❚❚  PAUSED")
        self._status.setProperty("state", "listening" if listening else "paused")
        _repolish(self._status)
        if listening:
            self.term.clear()  # back to listening: the note belonged to the question just answered
            self.term.clearFocus()
        else:
            self.term.setFocus()  # paused: type a note straight away, no mouse needed
        self._sync_fix()
        self._sync_help()

    def _sync_fix(self) -> None:
        """Correcting only makes sense while paused, with a term typed."""
        self._fix.setEnabled(not self._listening and bool(self.term.text().strip()))

    def _emit_retry(self) -> None:
        if self._fix.isEnabled():
            self.retry_requested.emit(self.term.text().strip())
        self.term.clearFocus()  # sent or not, Space must toggle again straight away

    def _sync_help(self, *_: object) -> None:
        """Say exactly what Space does right now: toggle, or type a space."""
        focused, typed = self.term.hasFocus(), bool(self.term.text())
        self._fix_help.setText(TYPING_HELP if focused and typed else READY_HELP if focused else FIX_HELP)
        self._fix_help.setProperty("typing", focused and typed)
        _repolish(self._fix_help)

    def _set_stt(self, text: str, ok: bool) -> None:
        self._stt.setText(text)
        self._stt.setProperty("ok", ok)
        _repolish(self._stt)

    def show_notice(self, text: str) -> None:
        self._notice.setText(text)
        self._notice_timer.start()
