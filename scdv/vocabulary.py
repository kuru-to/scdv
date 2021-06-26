"""Sub module."""
from collections import UserList
import itertools
from typing import Union, List

from scdv.word import Word


class Vocabulary(UserList):
    """語彙 class

    Note:
        * 語彙なので集合なのだが, `Document` class でインデックスから
            単語を参照するため, UserList を継承

    Args:
        lst_lst_words: 語彙に登録する文書ごとの単語. 初期化時に唯一にして `Word` クラス化
    """
    def __init__(self, lst_lst_word: List[List[str]]):
        set_word = set(itertools.chain.from_iterable(lst_lst_word))
        super().__init__([Word(str(w)) for w in set_word])

    def __repr__(self):
        return "Vocabulary<'{:}'>".format("', '".join([str(w) for w in self]))

    def remove(self, word: Union[str, Word]):
        """ 語彙から削除する

        Args:
            word: 削除したい単語
        """
        super().remove(Word(str(word)))
