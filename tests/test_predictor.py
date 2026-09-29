"""Word prediction for the gaze keyboard."""
import os
import tempfile
import unittest

from keyboard.predictor import WordPredictor


class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = WordPredictor()

    def test_prefix_completion(self):
        self.assertIn("WATER", self.p.suggest("I NEED W"))
        self.assertIn("NURSE", self.p.suggest("PLEASE CALL THE N"))
        self.assertEqual(self.p.suggest("MED")[0], "MEDICINE")

    def test_next_word_after_space(self):
        s = self.p.suggest("I ")
        self.assertIn("NEED", s)
        self.assertIn("WANT", s)

    def test_sentence_start(self):
        self.assertIn("I", self.p.suggest("I NEED WATER. "))

    def test_returns_four_uppercase(self):
        s = self.p.suggest("TH")
        self.assertEqual(len(s), 4)
        self.assertTrue(all(w.isupper() for w in s))

    def test_no_offensive_suggestions(self):
        for pre in ("F", "FU", "SH", "BI", "AS"):
            self.assertNotIn("FUCK", self.p.suggest(pre))
            self.assertNotIn("SHIT", self.p.suggest(pre))

    def test_learns_and_persists_user_words(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "vocab.json")
            p = WordPredictor(user_vocab_path=path)
            for _ in range(2):
                p.learn_word("physiotherapy", "my")
            self.assertEqual(p.suggest("MY PH")[0], "PHYSIOTHERAPY")
            p.save()
            p2 = WordPredictor(user_vocab_path=path)
            self.assertEqual(p2.suggest("MY PH")[0], "PHYSIOTHERAPY")

    def test_split(self):
        self.assertEqual(WordPredictor.split("I NEED WA"), ("need", "wa"))
        self.assertEqual(WordPredictor.split("I NEED "), ("need", ""))
        self.assertEqual(WordPredictor.split("HELLO. I"), ("", "i"))


if __name__ == "__main__":
    unittest.main()
