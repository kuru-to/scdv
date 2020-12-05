#!/usr/bin/env python

"""Tests for `word` package."""

import unittest

import numpy as np

from scdv.vocabulary import Vocabulary


class Test_Vocabulary(unittest.TestCase):
    """Tests for `word` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.lst_lst_word = [["apple", "banana"], ["corch", "banana", "empty", ""], ["apple"]]

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_get_Vocabulary(self):
        vocab = Vocabulary(self.lst_lst_word)
        lst_word_in_vocab = vocab.get_words()

        # 同じ単語が2つ存在しないか確認
        for word in lst_word_in_vocab:
            self.assertEqual(len([1 for word_in_vocab in lst_word_in_vocab if word == word_in_vocab]), 1)
        return


    

if __name__ == "__main__":
    unittest.main()
