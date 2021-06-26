"""Sub module."""
from collections import UserDict
import itertools
from typing import Union, List

from scdv.word import Word


class Vocabulary(UserDict):
    """語彙 class

    Note:
        * 語彙なので集合なのだが, `Document` インスタンス作成時に `str` から
            `Word` を参照するため, UserDist を継承

    Args:
        lst_lst_words: 語彙に登録する文書ごとの単語. 初期化時に唯一にして `Word` クラス化
    """
    def __init__(self, lst_lst_word: List[List[str]]):
        set_word = set(itertools.chain.from_iterable(lst_lst_word))
        super().__init__({str(Word(w)): Word(w) for w in set_word})

    def __repr__(self):
        repr_str = "', '".join([w for w in self.keys()])
        return "Vocabulary<'{:}'>".format(repr_str)

    def __delitem__(self, word: Union[str, Word]):
        """ 語彙から削除する

        Args:
            word: 削除したい単語
        """
        super().__delitem__(str(word))
