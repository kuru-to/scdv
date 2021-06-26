"""Sub module."""
from typing import List

import numpy as np
from scdv.vocabulary import Vocabulary


class Document:
    """文書 class

    Args:
        idx: document id
        lst_idx_words_in_vocab: `Vocabulary` class での単語のインデックス.
            この値をもとに単語のidf値, および単語ベクトルを取得する
    """
    def __init__(self, idx: int, lst_idx_words_in_vocab: List[int] = []):
        self._id = idx
        self._lst_idx_words_in_vocab = lst_idx_words_in_vocab

    @property
    def id(self):
        return self._id

    def __repr__(self):
        str_id = ",".join(self._lst_idx_words_in_vocab)
        return "Document< id : {0}, index : [{1}]>".format(self.id, str_id)

    def get_words(self, vocab: Vocabulary):
        return [vocab[idx] for idx in self._lst_idx_words_in_vocab]

    @property
    def mean_word_vector(self):
        return self._mean_word_vector

    @mean_word_vector.setter
    def mean_word_vector(self, mean_word_vector: np.array):
        self._mean_word_vector = mean_word_vector

    def set_sparce_mean_vector(self, threshold: float, vocab: Vocabulary):
        """平均ベクトルのスパース化

        Args:
            threshold 閾値. この値より絶対値が小さければ0にする

        Attributes:
            _sparce_mean_vector: SCDV で求めたい文書ベクトル
        """
        mean_vector = self.mean_word_vector(vocab)
        self._sparce_mean_vector = np.where(
            np.abs(mean_vector) < threshold, 0, mean_vector
        )

    @property
    def sparce_mean_vector(self):
        return self._sparce_mean_vector

    @sparce_mean_vector.setter
    def sparce_mean_vector(self, sparce_mean_vector: np.array):
        self._sparce_mean_vector = sparce_mean_vector
