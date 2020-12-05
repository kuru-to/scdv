"""Sub module."""
import numpy as np

from scdv.word import Word

class Vocabulary:
    def __init__(self, lst_lst_word):
        # 単語のリストに展開
        lst_word = []
        for lst_input in lst_lst_word:
            lst_word += lst_input

        self.words = [Word(word) for word in set(lst_word)]
        return

    def get_words(self):
        return self.words.copy()

    def remove_word(self, remove_word):
        self.words = list(filter(lambda x: not x.match(remove_word), self.words))
        return
    
    
    