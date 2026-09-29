"""
Full-screen gaze keyboard layout.

Screen is tiled edge-to-edge with NO gaps, so every gaze point (including one clamped
to the screen border) belongs to exactly one key -> the edges of the screen are usable
and a small calibration error near a border still lands on the border key.

    +--------------------------------------------------------------+-------+
    |  typed text                                                  | PAUSE |
    +------------+------------+------------+------------+------------------+
    | suggestion | suggestion | suggestion | suggestion |   SPEAK (TTS)    |
    +-------+-------+----------+------+------+----------+-----+------------+
    | WATER | FOOD  | WASHROOM | HELP | PAIN | MEDICINE | YES |  NO        |
    +-------+-------+-------+-------+-------+-------+-------+------------+
    |   A   |   B   |   C   |   D   |   E   |   F   |   G   |   H        |
    |   I   |   J   |   K   |   L   |   M   |   N   |   O   |   P        |
    |   Q   |   R   |   S   |   T   |   U   |   V   |   W   |   X        |
    |   Y   |   Z   |   .   |   ?   | SPACE | BACKSPACE | CLEAR | ENTER  |
    +--------------------------------------------------------------------+

Alphabetical order on purpose: for users who cannot use a physical keyboard, a
predictable order is easier to learn than QWERTY. Phrases are editable in
config/keyboard_phrases.json.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

DEFAULT_PHRASES = [
    ("WATER", "I need water"),
    ("FOOD", "I am hungry"),
    ("WASHROOM", "I need to use the washroom"),
    ("HELP", "Please help me"),
    ("PAIN", "I am in pain"),
    ("MEDICINE", "I need my medicine"),
    ("YES", "Yes"),
    ("NO", "No"),
]

# 8 x 4 grid -> near-square keys (240 x 194 px on 1920x1080). With webcam error now similar
# on both axes, the SMALLEST key dimension decides the hit rate; 8x4 beats 6x5 (320 x 162).
LETTER_ROWS = [
    ["A", "B", "C", "D", "E", "F", "G", "H"],
    ["I", "J", "K", "L", "M", "N", "O", "P"],
    ["Q", "R", "S", "T", "U", "V", "W", "X"],
    ["Y", "Z", ".", "?", "SPACE", "DEL", "CLEAR", "ENTER"],
]
CONTROL_KEYS = {"SPACE", "DEL", "CLEAR", "SPEAK", "PAUSE", "ENTER"}
KEY_LABELS = {"DEL": "BACKSPACE"}          # id stays "DEL" (config/tests/drift undo use it)

# Relative row heights: [suggestions, phrases, letters x3, bottom controls]. The bottom row is
# the hardest to hit with webcam gaze (lids cover the iris, camera looks down on the eye) and
# holds SPACE / BACKSPACE / ENTER, so it gets extra height, taken mostly from the phrase row.
ROW_WEIGHTS = [1.0, 0.9, 1.0, 1.0, 1.0, 1.2]
N_SUGGESTIONS = 4


@dataclass
class Key:
    id: str
    label: str
    kind: str            # "phrase" | "letter" | "control" | "suggestion" | "display"
    x: float
    y: float
    w: float
    h: float
    output: str = ""     # text inserted / spoken
    dwell_time: float = 1.0
    enabled: bool = True # empty suggestion slots are disabled (cannot be dwelled on)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


def load_phrases(path: Optional[str]) -> List[tuple]:
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            phrases = [(p["label"].upper(), p.get("speak", p["label"])) for p in data["phrases"]]
            if 2 <= len(phrases) <= 10:
                return phrases
        except Exception as e:
            print(f"[Keyboard] could not read {path}: {e} - using defaults")
    return list(DEFAULT_PHRASES)


def build_layout(screen_w: int, screen_h: int,
                 phrases: Optional[Sequence[tuple]] = None,
                 text_bar_frac: float = 0.08,
                 suggestion_row: bool = True,
                 letter_dwell: float = 1.0,
                 phrase_dwell: float = 1.2,
                 control_dwell: float = 1.2) -> List[Key]:
    phrases = list(phrases or DEFAULT_PHRASES)
    keys: List[Key] = []
    bar_h = screen_h * text_bar_frac
    weights = list(ROW_WEIGHTS) if suggestion_row else list(ROW_WEIGHTS[1:])
    unit = (screen_h - bar_h) / sum(weights)
    heights = [w * unit for w in weights]

    # text bar: display (not selectable) + PAUSE in the top-right corner
    pause_w = screen_w / 8
    keys.append(Key("DISPLAY", "", "display", 0, 0, screen_w - pause_w, bar_h))
    keys.append(Key("PAUSE", "PAUSE", "control", screen_w - pause_w, 0, pause_w, bar_h,
                    dwell_time=control_dwell))

    y0 = bar_h
    # word-suggestion row + SPEAK (text-to-speech) in the right-hand slot
    if suggestion_row:
        sw = screen_w / (N_SUGGESTIONS + 1)
        for i in range(N_SUGGESTIONS):
            keys.append(Key(f"SUG_{i}", "", "suggestion", i * sw, y0, sw, heights[0], dwell_time=letter_dwell))
        keys.append(Key("SPEAK", "SPEAK", "control", N_SUGGESTIONS * sw, y0, sw, heights[0],
                        dwell_time=control_dwell))
        y0 += heights.pop(0)

    # phrase row
    pw = screen_w / len(phrases)
    for i, (label, speak) in enumerate(phrases):
        keys.append(Key(f"PHRASE_{label}", label, "phrase", i * pw, y0, pw, heights[0],
                        output=speak, dwell_time=phrase_dwell))
    y0 += heights.pop(0)

    # letters + controls
    for r, row in enumerate(LETTER_ROWS):
        kw = screen_w / len(row)
        y, rh = y0, heights[r]
        y0 += rh
        for c, label in enumerate(row):
            is_ctrl = label in CONTROL_KEYS
            keys.append(Key(label, KEY_LABELS.get(label, label), "control" if is_ctrl else "letter",
                            c * kw, y, kw, rh,
                            output=" " if label == "SPACE" else ("" if is_ctrl else label),
                            dwell_time=control_dwell if is_ctrl else letter_dwell))
    return keys
