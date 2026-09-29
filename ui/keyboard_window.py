"""
Full-screen Gaze Keyboard window (word suggestions + text-to-speech + phrases + A-Z).

Fed by the dashboard loop exactly like the benchmark window:
    kb.feed_gaze(smooth_x, smooth_y, is_valid, raw_xy=(raw_x, raw_y))

  * Suggestion row: 4 predicted words, updated after every key. Selecting one completes the
    current word (or adds the next word) plus a space. Learns the user's own vocabulary.
  * SPEAK (text-to-speech): reads the typed text aloud, offline. Selecting SPEAK while it is
    talking stops the speech. Optionally every finished sentence (". ?") is spoken automatically.
  * Phrase keys speak immediately; HELP also plays an alert beep.
  * PAUSE (top-right) blocks all selections except PAUSE itself. ESC closes (operator).
"""

import os
import threading
import time
import tkinter as tk
from typing import Callable, Optional, Tuple

from keyboard.layout import build_layout, load_phrases
from keyboard.selector import KeySelector, SelectionEvent
from keyboard.predictor import WordPredictor
from keyboard.speech import Speaker

COLORS = {
    "bg": "#0F1117", "key": "#1E2333", "key_border": "#0F1117",
    "phrase": "#23304A", "control": "#2A2438", "help": "#5A1A24",
    "suggestion": "#1C3A33", "speak": "#1F4E2C", "disabled": "#161923",
    "active": "#2F6FEB", "progress": "#00D26A", "text": "#F2F4F8", "muted": "#6B7280",
    "display": "#12151F", "paused": "#7A5A00", "cursor": "#FFCC00",
}


def _beep_alert():
    """Audible alert for HELP: Windows beep, macOS system sound, else terminal bell."""
    def _run():
        import shutil, subprocess, sys
        for _ in range(3):
            try:
                if sys.platform.startswith("win"):
                    import winsound
                    winsound.Beep(1200, 250)
                elif sys.platform == "darwin" and shutil.which("afplay"):
                    subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"], timeout=3)
                else:
                    print("\a", end="", flush=True)
            except Exception:
                pass
            time.sleep(0.1)
    threading.Thread(target=_run, daemon=True).start()


class GazeKeyboardWindow(tk.Toplevel):
    def __init__(self, parent, screen_w: int, screen_h: int,
                 phrases_file: Optional[str] = None,
                 letter_dwell: float = 1.0, phrase_dwell: float = 1.2,
                 show_cursor: bool = True,
                 on_selection: Optional[Callable[[SelectionEvent], bool]] = None,
                 on_undo: Optional[Callable[[], None]] = None,
                 user_vocab_file: Optional[str] = None,
                 speak_on_sentence_end: bool = True,
                 speech_rate: int = 160,
                 speech_echo: str = "word",
                 speech_backend: str = "auto",
                 selector_params: Optional[dict] = None,
                 typed_log_file: Optional[str] = None):
        super().__init__(parent)
        self.title("Gaze Keyboard")
        self.attributes("-fullscreen", True)
        self.configure(bg=COLORS["bg"])
        self.screen_w, self.screen_h = screen_w, screen_h
        self.on_selection = on_selection
        self.on_undo = on_undo
        self.show_cursor = show_cursor
        self.speak_on_sentence_end = speak_on_sentence_end
        # speech_echo: "word"   = speak every word as soon as it is finished (default)
        #              "letter" = also speak every letter as it is typed
        #              "sentence" = only speak finished sentences / SPEAK key
        #              "off"    = only the SPEAK key and phrase keys talk
        self.speech_echo = speech_echo
        self.speaker = Speaker(rate_wpm=speech_rate, backend=speech_backend)
        self.predictor = WordPredictor(user_vocab_path=user_vocab_file)
        self.typed = ""
        self._history = []                     # stack of (typed_before, learned_from_selection)
        self._flash_until = {}

        self.keys = build_layout(screen_w, screen_h, load_phrases(phrases_file),
                                 letter_dwell=letter_dwell, phrase_dwell=phrase_dwell)
        self.sug_keys = [k for k in self.keys if k.kind == "suggestion"]
        self.selector = KeySelector(self.keys, screen_w, screen_h, **(selector_params or {}))
        self.typed_log_file = typed_log_file
        letter = next(k for k in self.keys if k.kind == "letter")
        self.letter_key_size = (letter.w, letter.h)      # used by the dashboard's gaze stabiliser

        self.canvas = tk.Canvas(self, bg=COLORS["bg"], highlightthickness=0,
                                width=screen_w, height=screen_h)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._items = {}
        self._draw_static()
        self._cursor = self.canvas.create_oval(0, 0, 0, 0, outline=COLORS["cursor"], width=3, state="hidden")
        self._refresh_suggestions()

        self.bind("<Escape>", lambda e: self.close())
        self.bind("<F2>", lambda e: self._toggle_cursor())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.focus_force()

    def close(self):
        try:
            self.predictor.save()
            self.speaker.close()
        finally:
            self.destroy()

    # ------------------------------------------------------------ drawing
    def _base_color(self, k):
        if k.id == "PHRASE_HELP":
            return COLORS["help"]
        if k.id == "SPEAK":
            return COLORS["speak"]
        if k.kind == "suggestion":
            return COLORS["suggestion"] if k.enabled else COLORS["disabled"]
        return {"phrase": COLORS["phrase"], "control": COLORS["control"]}.get(k.kind, COLORS["key"])

    def _font_size(self, k):
        if k.kind == "letter":
            return 44
        n = len(k.label)
        return 20 if n >= 11 else (22 if n >= 8 else (26 if n > 5 else 30))

    def _draw_static(self):
        for k in self.keys:
            if k.kind == "display":
                self.canvas.create_rectangle(k.x, k.y, k.x + k.w, k.y + k.h,
                                             fill=COLORS["display"], outline=COLORS["key_border"], width=4)
                self._display_txt = self.canvas.create_text(k.x + 24, k.cy, anchor="w", text="",
                                                            fill=COLORS["text"], font=("Segoe UI", 28, "bold"))
                # speech status (right side of the text bar) - makes a dead voice visible
                self._voice_txt = self.canvas.create_text(k.x + k.w - 16, k.cy, anchor="e", text="",
                                                          fill=COLORS["muted"], font=("Segoe UI", 12))
                continue
            rect = self.canvas.create_rectangle(k.x, k.y, k.x + k.w, k.y + k.h, fill=self._base_color(k),
                                                outline=COLORS["key_border"], width=6)
            bar = self.canvas.create_rectangle(k.x + 6, k.y + k.h - 14, k.x + 6, k.y + k.h - 6,
                                               fill=COLORS["progress"], outline="", state="hidden")
            txt = self.canvas.create_text(k.cx, k.cy, text=k.label, fill=COLORS["text"],
                                          font=("Segoe UI", self._font_size(k), "bold"))
            self._items[k.id] = (rect, bar, txt)

    def _toggle_cursor(self):
        self.show_cursor = not self.show_cursor

    def _refresh_suggestions(self):
        words = self.predictor.suggest(self.typed, n=len(self.sug_keys))
        for i, k in enumerate(self.sug_keys):
            w = words[i] if i < len(words) else ""
            k.label = w
            k.output = w
            k.enabled = bool(w)
            txt = self._items[k.id][2]
            self.canvas.itemconfig(txt, text=w, font=("Segoe UI", self._font_size(k), "bold"))

    def _render(self, gx, gy):
        st = self.selector.state
        now = time.time()
        for k in self.keys:
            if k.id not in self._items:
                continue
            rect, bar, _ = self._items[k.id]
            active = st.active_key is not None and st.active_key.id == k.id and k.enabled
            fill = COLORS["active"] if active and not self.selector.paused_by_user else self._base_color(k)
            if self._flash_until.get(k.id, 0) > now:
                fill = COLORS["progress"]
            if k.id == "PAUSE" and self.selector.paused_by_user:
                fill = COLORS["paused"]
            self.canvas.itemconfig(rect, fill=fill)
            p = st.progress if active else 0.0
            self.canvas.coords(bar, k.x + 6, k.y + k.h - 14, k.x + 6 + (k.w - 12) * p, k.y + k.h - 6)
            # amber bar = repeating the key that was just typed (takes longer on purpose)
            self.canvas.itemconfig(bar, state="normal" if p > 0 else "hidden",
                                   fill=COLORS["cursor"] if (active and st.repeat) else COLORS["progress"])
        self.canvas.itemconfig(self._items["PAUSE"][2], text="RESUME" if self.selector.paused_by_user else "PAUSE")
        speak_label = ("STOP" if self.speaker.is_speaking else "SPEAK") if self.speaker.available else "NO VOICE"
        self.canvas.itemconfig(self._items["SPEAK"][2], text=speak_label)
        if not self.speaker.available or self.speaker.last_error:
            self.canvas.itemconfig(self._voice_txt, text=self.speaker.status()[:70], fill="#FF6B6B")
        else:
            self.canvas.itemconfig(self._voice_txt, text=f"🔊 {self.speaker.backend} · echo: {self.speech_echo}",
                                   fill=COLORS["muted"])
        if self.show_cursor and gx is not None and gy is not None:
            r = 14
            self.canvas.coords(self._cursor, gx - r, gy - r, gx + r, gy + r)
            self.canvas.itemconfig(self._cursor, state="normal")
            self.canvas.tag_raise(self._cursor)
        else:
            self.canvas.itemconfig(self._cursor, state="hidden")
        shown = self.typed[-55:] + "▌"
        self.canvas.itemconfig(self._display_txt, text=("[PAUSED]  " if self.selector.paused_by_user else "") + shown)

    # ------------------------------------------------------------ speech
    def _say(self, text: str, interrupt: bool = True):
        """interrupt=True for SPEAK/phrases/sentences; word/letter echoes queue up instead."""
        text = text.strip()
        if not text:
            return
        if not self.speaker.available:
            print(f"[Keyboard] {self.speaker.status()}")
            return
        self.speaker.say(text.lower() if len(text) > 1 else text, interrupt=interrupt)

    def _echo_word(self, word: str):
        if self.speech_echo in ("word", "letter") and word:
            self._say(word, interrupt=False)

    def _echo_letter(self, letter: str):
        if self.speech_echo == "letter" and letter.strip():
            self._say(letter, interrupt=False)

    # ------------------------------------------------------------ input
    def feed_gaze(self, gx: Optional[float], gy: Optional[float], is_valid: bool,
                  raw_xy: Optional[Tuple[float, float]] = None,
                  stable: Optional[bool] = None, last_saccade_t: Optional[float] = None) -> None:
        if not self.winfo_exists():
            return
        ev = self.selector.update(gx, gy, is_valid, time.time(), raw_xy,
                                  stable=stable, last_saccade_t=last_saccade_t)
        if ev is not None:
            self._handle(ev)
            self._refresh_suggestions()
        self._render(gx, gy)

    def _handle(self, ev: SelectionEvent):
        k = ev.key
        before = self.typed
        learn = k.kind in ("letter", "phrase", "suggestion")   # controls are not used to learn drift
        if k.id == "PAUSE":
            return
        if k.id == "DEL":
            if self._history:
                prev, learned = self._history.pop()
                self.typed = prev
                if learned and self.on_undo:
                    self.on_undo()                  # drop the drift sample of the undone key
            return
        if k.id == "CLEAR":
            self._history.append((before, False))
            self.typed = ""
            return
        if k.id == "ENTER":
            # finish the line: speak it, keep a log for carers, start a new line (BACKSPACE undoes)
            line = self.typed.strip()
            if line:
                self._say(line)
                self._log_line(line)
                prev, partial = self.predictor.split(self.typed)
                if partial:
                    self.predictor.learn_word(partial, prev)
            self._history.append((before, False))
            self.typed = ""
            self._flash(k)
            return
        if k.id == "SPEAK":
            if self.speaker.is_speaking:
                self.speaker.stop()
            else:
                self._say(self.typed)
            return

        if k.kind == "phrase":
            self.typed = (self.typed + " " if self.typed and not self.typed.endswith(" ") else self.typed) + k.output.upper() + ". "
            self._say(k.output)
            if k.id == "PHRASE_HELP":
                _beep_alert()
        elif k.kind == "suggestion":
            if not k.enabled:
                return
            prev, partial = self.predictor.split(self.typed)
            base = self.typed[:len(self.typed) - len(partial)] if partial else self.typed
            if base and not base.endswith(" "):
                base += " "
            self.typed = base + k.output + " "
            self.predictor.learn_word(k.output, prev)
            self._echo_word(k.output)
        else:
            if k.output in (" ", ".", "?", ","):
                # a word just finished -> learn it and speak it
                prev, partial = self.predictor.split(self.typed)
                if partial:
                    self.predictor.learn_word(partial, prev)
                    if not (k.output in (".", "?") and self.speak_on_sentence_end):
                        self._echo_word(partial)      # sentence end speaks the whole sentence instead
            else:
                self._echo_letter(k.output)
            if k.output in (".", "?", ",") and self.typed.endswith(" "):
                self.typed = self.typed[:-1]         # "WORD ." -> "WORD."
            self.typed += k.output
            if k.output in (".", "?"):
                self.typed += " "
                if self.speak_on_sentence_end or self.speech_echo != "off":
                    self._say(self._last_sentence(self.typed), interrupt=False)
        learned = False
        if learn and self.on_selection:
            learned = bool(self.on_selection(ev))
        self._history.append((before, learned))
        self._flash(k)

    def _log_line(self, line: str):
        if not self.typed_log_file:
            return
        try:
            import datetime
            os.makedirs(os.path.dirname(os.path.abspath(self.typed_log_file)), exist_ok=True)
            with open(self.typed_log_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\t{line}\n")
        except Exception as e:
            print(f"[Keyboard] could not write typed log: {e}")

    @staticmethod
    def _last_sentence(text: str) -> str:
        t = text.rstrip()
        body = t[:-1] if t and t[-1] in ".?" else t
        cut = max(body.rfind("."), body.rfind("?"))
        return t[cut + 1:].strip()

    def _flash(self, k):
        self._flash_until[k.id] = time.time() + 0.25     # confirmation feedback
