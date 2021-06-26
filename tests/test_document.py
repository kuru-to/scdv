#!/usr/bin/env python

"""Tests for `document` package."""

import unittest

from scdv.vocabulary import Vocabulary
from scdv.document import Document


class TestScdv(unittest.TestCase):
    """Tests for `document` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        lst_lst_word = [
            ["apple", "banana"], ["corch", "banana", "empty", ""], ["apple"]
        ]
        self._vocab = Vocabulary(lst_lst_word)
        self._lst_word = lst_lst_word[0]
        # 0番目の文書をテスト対象とする
        lst_word_in_vocab = [
            self._vocab[word] for word in self._lst_word
        ]
        self._document = Document(0, lst_word_in_vocab)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_lst_word(self):
        """
        テスト項目:
            * テスト対象の文書の単語が単語リストの1番目のリストと一致すること
        """
        test_lst = self._document.lst_word
        self.assertEqual(len(test_lst), len(self._lst_word))
        for w in self._lst_word:
            self.assertIn(w, test_lst)
