"""
Offline word prediction for the gaze keyboard.

Every avoided letter saves the user ~1-1.5 s of dwell, so prediction is the single biggest
typing-speed lever for eye-gaze users.

Sources, combined into one score per candidate:
  1. Base English frequencies (keyboard/wordlist_en.txt, ~20k words, bundled - no internet).
  2. A care / AAC vocabulary boost (water, washroom, nurse, pain, ...).
  3. The user's own history (data/user_vocab.json): words and word pairs they actually type
     are learned and ranked higher next time. Nothing leaves the machine.
  4. Next-word prediction after a space: learned word pairs + a small built-in table of
     common care phrases ("I need ...", "I am ...", "please ...").

Offensive words from the frequency list are never suggested (a user can still type them).
"""

import bisect
import json
import math
import os
import re
import threading
from collections import Counter
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORDLIST = os.path.join(_HERE, "wordlist_en.txt")

CARE_WORDS = """
water drink food hungry thirsty eat washroom toilet bathroom urine pee poop bedpan diaper
help nurse doctor medicine tablet pain hurt ache headache stomach chest back leg arm hand
head neck breathe breathing cough cold hot fever sick tired sleep sleepy awake bed pillow
blanket turn sit up down left right move lift change clothes wash bath shower clean wet dry
light lights fan ac tv television phone call family mother father mom dad wife husband son
daughter brother sister friend home hospital please thank thanks yes no maybe stop wait
again more less enough finished done good bad better worse okay fine happy sad scared angry
bored lonely uncomfortable itchy itch glasses teeth mouth suction oxygen tube feed feeding
music read book window door open close quiet noise loud visit visitor today tomorrow now
later morning night time love miss want need like feel think know tell
""".split()

# built-in next-word table (lower-case). Learned pairs from the user always add on top.
NEXT_WORDS: Dict[str, List[str]] = {
    "": ["i", "please", "can", "thank", "yes", "no", "where", "when"],
    "i": ["need", "want", "am", "feel", "have", "love", "can't", "don't"],
    "need": ["water", "help", "to", "the", "my", "a", "medicine", "washroom"],
    "want": ["to", "water", "food", "my", "the", "some", "a", "sleep"],
    "am": ["in", "hungry", "thirsty", "tired", "cold", "hot", "fine", "okay"],
    "feel": ["sick", "tired", "cold", "hot", "pain", "better", "worse", "good"],
    "please": ["call", "help", "come", "turn", "give", "change", "open", "close"],
    "call": ["the", "my", "nurse", "doctor", "mom", "dad", "family", "me"],
    "my": ["family", "medicine", "phone", "glasses", "back", "head", "leg", "wife"],
    "to": ["the", "sleep", "eat", "drink", "go", "sit", "talk", "see"],
    "the": ["nurse", "doctor", "light", "tv", "fan", "door", "window", "bed"],
    "turn": ["me", "off", "on", "the", "left", "right", "over", "up"],
    "thank": ["you"],
    "can": ["you", "i", "we"],
    "you": ["please", "help", "come", "call", "give", "turn"],
    "in": ["pain", "the", "my", "bed"],
    "have": ["pain", "a", "to", "some"],
}

_BLOCK = set("""
fuck fucking fucked fucker fuckin shit shitty bitch bitches bastard asshole ass dick dicks
cock pussy cunt slut whore porn porno nigga nigger fag faggot retard retarded rape raped
sex sexy tits boobs penis vagina cum horny milf wtf damn crap piss
""".split())

_WORD_RE = re.compile(r"[a-z']+")


class WordPredictor:
    def __init__(self, wordlist_path: Optional[str] = DEFAULT_WORDLIST,
                 user_vocab_path: Optional[str] = None, max_words: int = 20000):
        self.base: Dict[str, float] = {}
        if wordlist_path and os.path.exists(wordlist_path):
            with open(wordlist_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or "\t" not in line:
                        continue
                    w, z = line.rstrip("\n").split("\t")
                    if w not in _BLOCK:
                        self.base[w] = float(z)
                    if len(self.base) >= max_words:
                        break
        for w in CARE_WORDS:
            self.base.setdefault(w, 3.0)
        for nxt in NEXT_WORDS.values():
            for w in nxt:
                self.base.setdefault(w, 3.0)
        self._care = set(CARE_WORDS)
        self.user_vocab_path = user_vocab_path
        self.uni: Counter = Counter()
        self.bi: Counter = Counter()
        self._lock = threading.Lock()
        self._load_user()
        self._rebuild_index()

    # ------------------------------------------------------------- persistence
    def _load_user(self):
        if self.user_vocab_path and os.path.exists(self.user_vocab_path):
            try:
                with open(self.user_vocab_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.uni.update(d.get("unigrams", {}))
                self.bi.update(d.get("bigrams", {}))
            except Exception as e:
                print(f"[Predictor] could not read user vocab: {e}")

    def save(self):
        if not self.user_vocab_path:
            return
        with self._lock:
            data = {"unigrams": dict(self.uni.most_common(5000)),
                    "bigrams": dict(self.bi.most_common(10000))}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.user_vocab_path)), exist_ok=True)
            tmp = self.user_vocab_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.user_vocab_path)
        except Exception as e:
            print(f"[Predictor] could not save user vocab: {e}")

    def _rebuild_index(self):
        self._sorted = sorted(set(self.base) | set(self.uni))

    # ------------------------------------------------------------------ scoring
    def _score(self, w: str, prev: str) -> float:
        s = self.base.get(w, 2.0)
        if w in self._care:
            s += 1.5
        s += 3.0 * math.log1p(self.uni.get(w, 0))                 # the user's own words win quickly
        s += 5.0 * math.log1p(self.bi.get(f"{prev} {w}", 0))      # ... especially in the same context
        if w in NEXT_WORDS.get(prev, ()):
            s += 3.0 - 0.2 * NEXT_WORDS[prev].index(w)
        return s

    @staticmethod
    def split(text: str):
        """-> (previous complete word, current partial word) in lower case."""
        t = text.lower()
        partial = ""
        m = re.search(r"[a-z']+$", t)
        if m:
            partial = m.group(0)
            t = t[:m.start()]
        # previous word only counts inside the same sentence
        tail = re.split(r"[.?!]", t)[-1]
        words = _WORD_RE.findall(tail)
        prev = words[-1] if words else ""
        return prev, partial

    def suggest(self, text: str, n: int = 4) -> List[str]:
        prev, partial = self.split(text)
        if partial:
            lo = bisect.bisect_left(self._sorted, partial)
            hi = bisect.bisect_left(self._sorted, partial + "\x7f")
            cands = self._sorted[lo:hi]
            if len(cands) > 4000:                      # 1-letter prefix: pre-filter by frequency
                cands = sorted(cands, key=lambda w: -self.base.get(w, 0))[:600]
            cands = [w for w in cands if w != partial or len(w) > 1]
        else:
            learned = [k.split(" ", 1)[1] for k, _ in self.bi.most_common() if k.split(" ", 1)[0] == prev][:20]
            cands = list(dict.fromkeys(learned + NEXT_WORDS.get(prev, NEXT_WORDS[""] if not prev else [])
                                       + (NEXT_WORDS[""] if not prev else [])))
            if len(cands) < n:
                cands += [w for w in ("the", "to", "is", "please", "and", "my", "a", "it") if w not in cands]
        ranked = sorted(cands, key=lambda w: -self._score(w, prev))
        return [w.upper() for w in ranked[:n]]

    # ----------------------------------------------------------------- learning
    def learn_text(self, text: str):
        """Learn every complete word + word pair in text (call when a word is finished)."""
        for sentence in re.split(r"[.?!]", text.lower()):
            words = _WORD_RE.findall(sentence)
            prev = ""
            for w in words:
                with self._lock:
                    self.uni[w] += 1
                    self.bi[f"{prev} {w}"] += 1
                prev = w
        self._rebuild_index()

    def learn_word(self, word: str, prev: str = ""):
        w = word.lower().strip()
        if not _WORD_RE.fullmatch(w):
            return
        with self._lock:
            new = w not in self.uni and w not in self.base
            self.uni[w] += 1
            self.bi[f"{prev.lower()} {w}"] += 1
        if new:
            self._rebuild_index()
