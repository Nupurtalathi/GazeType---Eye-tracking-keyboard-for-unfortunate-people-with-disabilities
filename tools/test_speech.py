"""
Text-to-speech self-test. Run this FIRST if the keyboard does not talk:

    python tools/test_speech.py              # auto-select backend
    python tools/test_speech.py --backend say
    python tools/test_speech.py --list       # show which backends this machine has

It speaks three sentences through the same Speaker class the keyboard uses and prints
the backend, timing and any error. Check the system volume / output device if it reports
success but you hear nothing.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyboard.speech import Speaker   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--rate", type=int, default=160)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for b in ("say", "sapi", "powershell", "espeak-ng", "espeak", "spd-say", "pyttsx3"):
            sp = Speaker(backend=b)
            print(f"  {b:11s} {'OK' if sp.available else 'not available'}  {sp.last_error}")
            sp.close()
        return
    sp = Speaker(rate_wpm=a.rate, backend=a.backend)
    print("platform:", sys.platform, "|", sp.status())
    if not sp.available:
        print("-> no voice. macOS: `say` should always exist. Windows: pip install comtypes. "
              "Linux: sudo apt install espeak-ng")
        sys.exit(1)
    for text in ("Text to speech test.", "I need water.", "Please help me."):
        t0 = time.time()
        sp.say(text)
        time.sleep(0.2)
        while sp.is_speaking and time.time() - t0 < 15:
            time.sleep(0.05)
        print(f"  spoke {text!r:24s} in {time.time() - t0:4.1f}s  {('ERROR ' + sp.last_error) if sp.last_error else 'ok'}")
    print("stop test: starting a long sentence and stopping after 1 s ...")
    sp.say("This is a long sentence that should be cut off after about one second of speech.")
    time.sleep(1.0)
    sp.stop()
    time.sleep(0.3)
    print("  stopped:", not sp.is_speaking)
    sp.close()


if __name__ == "__main__":
    main()
