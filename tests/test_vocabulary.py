#!/usr/bin/env python

"""Tests for `vocabulary` module."""

import unittest
import itertools

from scdv.vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):
    """Tests for `Vocabulary` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self._lst_lst_word = [
            ["a", "b"], ["c", "d", "e", ""], ["a"]
        ]

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_remove(self):
        """
        テスト項目:
            * 削除したい単語が消えているか
            * 削除したくない単語が消えていないか
        """
        remove_word = "a"

        vocab = Vocabulary(self._lst_lst_word)
        vocab.remove(remove_word)

        set_answer = set(itertools.chain.from_iterable(
            self._lst_lst_word
        ))
        set_answer.remove(remove_word)

        self.assertNotIn(remove_word, vocab)

        for ans in set_answer:
            self.assertIn(ans, vocab)


if __name__ == "__main__":
    unittest.main()
