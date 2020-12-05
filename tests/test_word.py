#!/usr/bin/env python

"""Tests for `word` package."""

import unittest

import numpy as np

from scdv.word import Word


class TestWord(unittest.TestCase):
    """Tests for `word` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    # Word.match のテスト
    def test_Word_match(self):
        word="A"
        model = Word(word)
        self.assertTrue(model.match(word))
        
        # 異なる場合
        word_notMatch = "b"
        self.assertFalse(model.match(word_notMatch))
        return

    # Word の get_name のテスト
    def test_Word_get_name(self):
        name = "a"
        word = Word(name)
        self.assertTrue(word.get_name(), name)
        return

    # set_idf のテスト
    def test_Word_set_idf(self):
        name = "apple"
        idf = 0.3
        
        word = Word(name)
        word.set_idf(idf)
        
        self.assertEqual(word.get_idf(), idf)
        return

    # calc_clustered_vector のテスト
    def test_Word_calc_clustered_vector(self):
        name = "apple"
        idf = 0.3
        vector = np.array([2,4,2,1])
        cluster_idx = 2
        cluster_probability = np.array([0.1, 0.7, 0.2])
        
        word = Word(name)
        word.set_vector(vector)
        word.set_idf(idf)
        word.set_cluster_idx(cluster_idx)
        word.set_cluster_probability(cluster_probability)
        
        clustered_vector = word.calc_clustered_vector()
        
        self.assertEqual(clustered_vector.shape[0], vector.shape[0]*cluster_probability.shape[0])
        # 各値が等しいか確認する
        for idx_cluster_probability, prob in enumerate(cluster_probability):
            for idx_vector, value in enumerate(vector):
                test_value = clustered_vector[idx_vector+idx_cluster_probability*vector.shape[0]]
                true_value = idf*value*prob
                self.assertEqual(test_value, true_value)
        return

if __name__ == "__main__":
    unittest.main()
