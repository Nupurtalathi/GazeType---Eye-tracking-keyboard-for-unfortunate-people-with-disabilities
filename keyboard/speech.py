"""
Offline text-to-speech, reliable from a background thread (v5).

Why v4 was silent
-----------------
v4 tried pyttsx3 first and ran it in a worker thread. That is fragile:
  * macOS: pyttsx3's NSSpeechSynthesizer driver needs the main thread's run loop; in a worker
    thread runAndWait() never returns or says nothing. The worker then stayed "busy" forever,
    so SPEAK behaved as STOP and every later sentence was dropped - silently.
  * Windows: pyttsx3's SAPI5 driver needs COM initialised in the calling thread; without it
    engine creation fails, or only the first utterance plays.
Errors only went to the console, so nothing on screen showed the voice was dead.

v5 backends (first that works is used; override with TTS_BACKEND env var or config):
  macOS    : `say`                                   (built in)
  Windows  : SAPI.SpVoice via comtypes / pywin32, COM initialised in the worker thread
             -> fallback: PowerShell System.Speech  (built in)
  Linux    : espeak-ng / espeak / spd-say
  anywhere : pyttsx3 (last resort only)
All local - no text or audio leaves the machine. Every failure is recorded in
`last_error` and shown on the keyboard, and the worker can never stay stuck (watchdog).
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import List, Optional


class Speaker:
    WATCHDOG_BASE_S = 8.0          # max time allowed for one utterance: base + per-char
    WATCHDOG_PER_CHAR_S = 0.12

    def __init__(self, rate_wpm: int = 160, backend: Optional[str] = None, voice: Optional[str] = None):
        self.rate = int(rate_wpm)
        self.voice = voice
        self.requested = (backend or os.environ.get("TTS_BACKEND") or "auto").lower()
        self.backend = "none"
        self.last_error = ""
        self.spoken_count = 0
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._busy_since: Optional[float] = None
        self._proc: Optional[subprocess.Popen] = None
        self._sapi = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="tts", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=8.0)

    # ------------------------------------------------------------ public API
    @property
    def available(self) -> bool:
        return self.backend != "none"

    @property
    def is_speaking(self) -> bool:
        return self._busy_since is not None or not self._q.empty()

    def say(self, text: str, interrupt: bool = False) -> None:
        """Queue text. interrupt=True cuts off current speech first (use for SPEAK / phrases)."""
        text = " ".join((text or "").split())
        if not text:
            return
        if interrupt:
            self.stop()
        self._q.put(text)

    def stop(self) -> None:
        """Stop current speech and drop anything queued. Safe from any thread."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._stop_evt.set()
        p = self._proc
        if p is not None and p.poll() is None:
            self._kill(p)

    @staticmethod
    def _kill(p: subprocess.Popen) -> None:
        """Terminate the engine process *and* its children (e.g. shell wrappers)."""
        try:
            if os.name != "nt":
                import signal
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            else:
                p.terminate()
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass

    def status(self) -> str:
        if not self.available:
            return f"NO VOICE ({self.last_error or 'no engine found'})"
        return f"voice: {self.backend}" + (f" | last error: {self.last_error}" if self.last_error else "")

    # --------------------------------------------------------------- backends
    def _candidates(self) -> List[str]:
        if self.requested != "auto":
            return [self.requested]
        if sys.platform == "darwin":
            return ["say", "pyttsx3"]
        if sys.platform.startswith("win"):
            return ["sapi", "powershell", "pyttsx3"]
        return ["espeak-ng", "espeak", "spd-say", "pyttsx3"]

    def _try_init(self, name: str) -> bool:
        try:
            if name == "say":
                return shutil.which("say") is not None
            if name in ("espeak-ng", "espeak", "spd-say"):
                return shutil.which(name) is not None
            if name == "powershell":
                return shutil.which("powershell") is not None or shutil.which("powershell.exe") is not None
            if name == "sapi":
                return self._init_sapi()
            if name == "pyttsx3":
                import pyttsx3
                self._pyttsx3 = pyttsx3.init()
                self._pyttsx3.setProperty("rate", self.rate)
                return True
        except Exception as e:
            self.last_error = f"{name}: {e}"
        return False

    def _init_sapi(self) -> bool:
        voice = None
        try:
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()                     # COM must be initialised in THIS thread
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
        except Exception as e1:
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                voice = win32com.client.Dispatch("SAPI.SpVoice")
            except Exception as e2:
                self.last_error = f"sapi: {e1} / {e2}"
                return False
        voice.Rate = max(-10, min(10, int(round((self.rate - 160) / 18))))
        self._sapi = voice
        return True

    def _speak_one(self, text: str) -> None:
        b = self.backend
        if b == "sapi":
            SVSF_ASYNC, SVSF_PURGE = 1, 2
            self._sapi.Speak(text, SVSF_ASYNC)
            deadline = time.time() + self.WATCHDOG_BASE_S + self.WATCHDOG_PER_CHAR_S * len(text)
            while not self._sapi.WaitUntilDone(50):
                if self._stop_evt.is_set() or time.time() > deadline:
                    self._sapi.Speak("", SVSF_ASYNC | SVSF_PURGE)
                    break
            return
        if b == "pyttsx3":
            self._pyttsx3.say(text)
            self._pyttsx3.runAndWait()
            return
        cmd = self._command(text)
        flags = 0x08000000 if os.name == "nt" else 0     # CREATE_NO_WINDOW
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                      creationflags=flags, start_new_session=(os.name != "nt"))
        deadline = time.time() + self.WATCHDOG_BASE_S + self.WATCHDOG_PER_CHAR_S * len(text)
        while self._proc.poll() is None:
            if self._stop_evt.is_set() or time.time() > deadline:
                self._kill(self._proc)
                break
            time.sleep(0.03)
        try:
            self._proc.wait(timeout=1.0)
        except Exception:
            pass
        err = b""
        try:
            if self._proc.returncode is not None and self._proc.returncode > 0:   # exited on its own with an error
                err = self._proc.stderr.read() or b""
        finally:
            self._proc.stderr.close()
        if self._proc.returncode not in (0, None) and not self._stop_evt.is_set() and self._proc.returncode > 0:
            self.last_error = f"{b} exit {self._proc.returncode}: {err.decode(errors='ignore').strip()[:120]}"

    def _command(self, text: str) -> List[str]:
        b = self.backend
        if b == "say":
            cmd = ["say", "-r", str(self.rate)]
            if self.voice:
                cmd += ["-v", self.voice]
            return cmd + [(" " + text) if text.startswith("-") else text]
        if b == "powershell":
            safe = text.replace("'", "''")
            rate = max(-10, min(10, int(round((self.rate - 160) / 18))))
            ps = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.Rate = {rate}; $s.Speak('{safe}')")
            return ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps]
        if b in ("espeak-ng", "espeak"):
            return [b, "-s", str(self.rate), "--", text]
        if b == "spd-say":
            return ["spd-say", "-w", "--", text]
        raise RuntimeError(f"unknown backend {b}")

    # ----------------------------------------------------------------- worker
    def _worker(self):
        for name in self._candidates():
            if self._try_init(name):
                self.backend = name
                self.last_error = ""
                break
        if self.backend == "none":
            print(f"[Speech] no text-to-speech engine available: {self.last_error}")
        else:
            print(f"[Speech] using backend: {self.backend}")
        self._ready.set()
        while True:
            text = self._q.get()
            if text is None:
                return
            if not self.available:
                continue
            self._stop_evt.clear()
            self._busy_since = time.time()
            try:
                self._speak_one(text)
                self.spoken_count += 1
            except Exception as e:
                self.last_error = f"{self.backend}: {e}"
                print(f"[Speech] {self.last_error}")
            finally:
                self._busy_since = None
                self._proc = None

    def close(self):
        self.stop()
        self._q.put(None)
