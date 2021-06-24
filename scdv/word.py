"""Sub module."""
import numpy as np
from collections import UserString


class Word(UserString):
    """Word クラス

    Args:
        word: 単語
    """
    def __init__(self, word: str):
        self.name = word.lower()
        self._word = word.lower()

    def __str__(self):
        return self._word

    def __eq__(self, word):
        return self._word == word.lower()

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vec: np.array):
        self._vector = vec

    @property
    def cluster_idx(self):
        """所属確率最大の cluster のインデックス"""
        return self._cluster_idx

    @cluster_idx.setter
    def cluster_idx(self, cluster_idx: int):
        self._cluster_idx = cluster_idx

    @property
    def cluster_probability(self):
        """各clusterへの所属確率をセットする"""
        return self._cluster_probability

    @cluster_probability.setter
    def cluster_probability(self, probability: np.array):
        self._cluster_probability = probability

    @property
    def idf(self):
        return self._idf

    @idf.setter
    def idf(self, idf: float):
        self._idf = idf

    @property
    def clustered_vector(self):
        """各 cluster への所属確率にidf値をかけたものをconcatnateしたベクトル"""
        output = np.array([
            self.idf * clustered_prob * self.vector
            for clustered_prob in self.cluster_probability
        ]).flatten()
        return output
