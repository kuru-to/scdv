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
        self._vocab = Vocabulary(self._lst_lst_word)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_del(self):
        """
        テスト項目:
            * 削除したい単語が消えているか
            * 削除したくない単語が消えていないか
        """
        del_word = "a"

        del self._vocab[del_word]

        set_answer = set(itertools.chain.from_iterable(
            self._lst_lst_word
        ))
        set_answer.remove(del_word)

        self.assertNotIn(del_word, self._vocab.keys())

        for ans in set_answer:
            self.assertIn(ans, self._vocab.keys())


if __name__ == "__main__":
    unittest.main()
