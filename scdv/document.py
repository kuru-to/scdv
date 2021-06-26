"""Sub module."""
from typing import List

import numpy as np

from scdv.word import Word


class Document:
    """文書 class

    Args:
        idx: document id
        lst_word: 単語リスト. idf値, 単語ベクトルが格納されている
            この値をもとに単語のidf値, および単語ベクトルを取得する
    """
    def __init__(self, idx: int, lst_word: List[Word]):
        self._id = idx
        self._lst_word = lst_word

    @property
    def id(self):
        return self._id

    @property
    def lst_word(self):
        return self._lst_word

    def __repr__(self):
        str_id = ",".join(self._lst_word)
        return "Document< id : {0}, index : [{1}]>".format(self.id, str_id)

    def calc_mean_word_vector(self):
        """文書中の各単語のベクトルの平均を取る
        """
        lst_word = self.lst_word
        count = len(lst_word)
        sumWordVector = lst_word[0].clustered_vector
        for word in lst_word[1:]:
            sumWordVector += word.clustered_vector
        return sumWordVector / count

    @property
    def mean_word_vector(self):
        if not hasattr(self, "_mean_word_vector"):
            self._mean_word_vector = self.calc_mean_word_vector()
        return self._mean_word_vector

    @property
    def sparce_mean_vector(self):
        return self._sparce_mean_vector

    @sparce_mean_vector.setter
    def sparce_mean_vector(self, sparce_mean_vector: np.array):
        self._sparce_mean_vector = sparce_mean_vector
