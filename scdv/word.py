"""Sub module."""
import numpy as np

# Word クラス
class Word():
    # コンストラクタ
    # 入力
    # word str型の単語
    def __init__(self, word):
        self.name = word.lower()
        return
    
    # 単語のstr型を返す
    def get_name(self):
        return self.name
    
    # 与えられた文字列が一致するか確認する method
    def match(self, word):
        return self.name == word.lower()
    
    # word2vec で出力した vector をセットする
    # 入力
    # vector 数値ベクトル.リスト型でも可能
    def set_vector(self, vector):
        self.vector = np.array(vector)
        return
    
    # vector を出力する
    def get_vector(self):
        return np.array(self.vector)
    
    # 所属確率最大の cluster のインデックスをセットする
    def set_cluster_idx(self, idx):
        self.cluster_idx = idx
        return
    
    # cluster_idx を出力する
    def get_cluster_idx(self):
        return self.cluster_idx
    
    # 各clusterへの所属確率をセットする
    # 入力
    # probability numpy型
    def set_cluster_probability(self, probability):
        self.cluster_probability = probability
        return
    
    # cluster_probability を出力する
    def get_cluster_probability(self):
        return self.cluster_probability
    
    # idf 値をセットする
    def set_idf(self, idf):
        self.idf = idf
        return
    
    # idf 値を出力する
    def get_idf(self):
        return self.idf
    
    # 各 cluster への所属確率にidf値をかけたものをconcatnateしたベクトルを出力する
    # 入力
    # lst_clustered_prob 各クラスターへの所属確率のリスト
    def calc_clustered_vector(self):        
        # 各 cluster への所属確率にidf値をかけてconcatenate
        clustered_vector = np.array([self.get_idf()*clustered_prob*self.get_vector() for clustered_prob in self.get_cluster_probability()]).flatten()
        
        return clustered_vector
    
    # clustered_vector のセット
    def set_clustered_vector(self, clustered_vector):
        self.clustered_vector = clustered_vector.copy()
        return
    
    # clustered_vector の取得
    def get_clustered_vector(self):
        return self.clustered_vector