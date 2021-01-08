"""Sub module."""
import numpy as np

# Dpcument クラス
class Document:
    # コンストラクタ
    # 入力
    # idx document id
    # lst_word document に格納されている単語のリスト.str型
    def __init__(self, idx, lst_word):
        self.id = idx
        self.words = lst_word
        return
    
    # idを返す
    def get_id(self):
        return self.id
    
    # 単語のリストを返す
    def get_words(self):
        return [word.name for word in self.words]
    
    # 指定した単語がDocumentに存在するか確認する
    # 入力
    # word str型の単語
    def isExist_word(self, word):
        for word_class in self.words:
            if word_class.match(word):
                return True
        return False
    
    # 所属する各単語を平均化する
    def calc_meanWordVector(self):
        count = 1
        sumWordVector = self.words[0].get_clustered_vector()
        for word in self.words[1:]:
            sumWordVector = sumWordVector + word.get_clustered_vector()
            count += 1
        
        return sumWordVector / count
    
    # 所属する各単語の平均ベクトルをセット
    def set_meanWordVector(self, vec):
        self.meanWordVector = vec
        return
    
    # 平均ベクトルを取得
    def get_meanWordVector(self):
        return self.meanWordVector
    
    # 平均ベクトルをスパース化する
    # 入力
    # threshold 閾値.この値より絶対値が小さければ0にする
    def calc_sparceMeanVector(self, threshold):
        meanVector = self.get_meanWordVector()
        return np.where(np.abs(meanVector)< threshold, 0, meanVector)
    
    # sparce vector をセット
    def set_sparceMeanVector(self, vec):
        self.sparceMeanVector = vec
        return
    
    # sparce vector を取得
    def get_sparceMeanVector(self):
        return self.sparceMeanVector
    
    