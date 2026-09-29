"""Speaker must never go silent: queueing, interrupt, stop and watchdog, with a fake engine."""
import os
import stat
import tempfile
import time
import unittest

from keyboard.speech import Speaker


def _fake_say(dirpath, sleep_s):
    p = os.path.join(dirpath, "say")
    log = os.path.join(dirpath, "say.log")
    with open(p, "w") as f:
        f.write(f"#!/bin/sh\necho \"$@\" >> {log}\nsleep {sleep_s}\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return log


@unittest.skipIf(os.name == "nt", "uses a POSIX shell script as fake engine")
class TestSpeaker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_path = os.environ["PATH"]

    def tearDown(self):
        os.environ["PATH"] = self.old_path
        self.tmp.cleanup()

    def _speaker(self, sleep_s):
        log = _fake_say(self.tmp.name, sleep_s)
        os.environ["PATH"] = self.tmp.name + os.pathsep + self.old_path
        return Speaker(backend="say"), log

    def _wait_idle(self, sp, timeout=5):
        t0 = time.time()
        while sp.is_speaking and time.time() - t0 < timeout:
            time.sleep(0.02)

    def _lines(self, log):
        return open(log).read().splitlines() if os.path.exists(log) else []

    def test_every_word_is_spoken_in_order(self):
        sp, log = self._speaker(0.05)
        for w in ("i", "need", "water"):
            sp.say(w)                       # default: queue, do not interrupt
        time.sleep(0.1)
        self._wait_idle(sp)
        self.assertEqual([l.split()[-1] for l in self._lines(log)], ["i", "need", "water"])
        self.assertEqual(sp.spoken_count, 3)
        sp.close()

    def test_interrupt_drops_queue_and_speaks_new_text(self):
        sp, log = self._speaker(0.3)
        for w in ("one", "two", "three", "four"):
            sp.say(w)
        time.sleep(0.05)
        sp.say("help me", interrupt=True)
        time.sleep(0.1)
        self._wait_idle(sp)
        spoken = [l.split("160 ")[-1] for l in self._lines(log)]
        self.assertEqual(spoken[-1], "help me")
        self.assertNotIn("four", spoken)
        sp.close()

    def test_watchdog_frees_a_hung_engine(self):
        sp, log = self._speaker(60)          # engine that never finishes
        sp.WATCHDOG_BASE_S, sp.WATCHDOG_PER_CHAR_S = 0.5, 0.0
        sp.say("stuck")
        time.sleep(0.1)
        self.assertTrue(sp.is_speaking)
        self._wait_idle(sp, timeout=3)
        self.assertFalse(sp.is_speaking)     # v4 stayed "speaking" forever -> SPEAK acted as STOP
        sp.say("next")
        time.sleep(0.1)
        self.assertIn("next", open(log).read())
        sp.close()

    def test_stop(self):
        sp, _ = self._speaker(5)
        sp.say("long sentence")
        time.sleep(0.2)
        sp.stop()
        time.sleep(0.2)
        self.assertFalse(sp.is_speaking)
        sp.close()

    def test_missing_engine_reports_error(self):
        sp = Speaker(backend="definitely-not-a-tts")
        self.assertFalse(sp.available)
        self.assertIn("NO VOICE", sp.status())
        sp.close()


if __name__ == "__main__":
    unittest.main()
