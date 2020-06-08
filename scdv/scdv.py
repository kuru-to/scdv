"""Main module."""
## create word2vec
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.mixture import GaussianMixture

class SCDV():
    # ドキュメントのリスト
    documents = []
    # vocabraly のリスト
    vocabraly = []
    
    def __init__(self
                 , num_cluster=60 
                 , random_seed=0
                 , threshold=0.01
                 , embedding_dimension=200 # Word vector dimensionality
                 , max_iter = 50           # Gausian Mixture Model 作成時の反復回数最大値
                ):
        self.num_cluster = num_cluster
        self.random_seed = random_seed
        self.threshold = threshold
        self.embedding_dimension = embedding_dimension
        self.max_iter = max_iter
        
        return
    
    # 語彙のセッティング
    # 入力
    # lst_lst_word 語彙となる単語のリストのリスト（文書ごと）
    def set_vocabulary(self, lst_lst_word):
        # 展開
        lst_word = []
        for lst_input in lst_lst_word:
            lst_word += lst_input
        
        # 同じ単語が登録されないようにset型に変換してから登録する
        self.vocabulary = [Word(word) for word in set(lst_word)]
        return
    
    # vocabulary の取得
    # str型のリストで出力する
    def get_vocabulary(self):
        return [word.get_name() for word in self.vocabulary]
    
    # vocabularyから削除する
    # word 削除する単語.str型
    def remove_vocabulary(self, word):
        self.vocabulary = list(filter(lambda x: not x.match(word), self.vocabulary))
        return
        
    # making word2vec
    # Input
    # lst_lst_word 各Documentごとに1つのリストに単語をまとめたもののリスト. デフォルトでは空なので、Documentからとってくる
    def make_word2VecModel(self,
                        lst_lst_word,
                        min_word_count = 0,   # Minimum word count
                        num_workers = 1,      # Number of threads to run in parallel
                        context = 1,          # Context window size
                        downsampling = 1e-3    # Downsample setting for frequent words
                        ):
        
        # 各Documentごとの単語のリストが与えられているので、そちらを使用する
        sentences = lst_lst_word.copy()

        print ("Training word2Vec model...")
        # Train Word2Vec model.
        self.word2vec = Word2Vec(sentences, workers=num_workers, hs = 0, sg = 1, negative = 10, iter = 25,\
                    size=self.embedding_dimension, min_count = min_word_count, \
                    window = context, sample = downsampling, seed=self.random_seed)

        # L2ノルムで正規化.メモリが足りない場合は replace=True
        self.word2vec.init_sims(replace=True)
        
        # Get wordvectors for all words in vocabulary.
        return
    
    # word2Vec のモデルを取得する
    def get_word2VecModel(self):
        try:
            word2VecModel = self.word2vec
        except:
            assert False, "word2vec model has not been made yet."
        return word2VecModel
        
    # word2vec のモデルからベクトルを抽出して語彙中の各単語に格納
    def set_word2Vec(self):
        for word in self.vocabulary:
            # word vector が作成されていれば格納
            try:
                word_vector = self.get_word2VecModel()[word.name]
                word.set_vector(self.get_word2VecModel()[word.name])
            # word vector が作成されていない単語であれば、vocabulary から remove
            except:
                self.remove_vocabulary(word.name)
        return
    
    # vocabulary からベクトルを出力する
    def get_word2Vec(self):
        word_vectors = np.array(self.vocabulary[0].get_vector())
        word_vectors = word_vectors.reshape([1,word_vectors.shape[0]])
        for word in self.vocabulary[1:]:
            word_vectors = np.concatenate([word_vectors, word.get_vector().reshape([1,word_vectors.shape[1]])], axis=0)
        return word_vectors
    
    # clustering の値を算出
    def make_clusterModel(self):
        # 作成したモデルからとってくる
        word_vectors = self.get_word2Vec()
            
        # Initalize a GMM object and use it for clustering.
        clf =  GaussianMixture(n_components=self.num_cluster,
                        covariance_type="tied", init_params='kmeans', max_iter=self.max_iter)
        
        # Get cluster assignments.
        print ("Training clustering model...")
        clf.fit(word_vectors)
        
        # モデルの格納
        self.cluster_model = clf
        return
    
    # clustering のモデルを取得する
    def get_clusterModel(self):
        try:
            clusterModel = self.cluster_model
        except:
            assert False, "clustering model has not been made yet."
        return clusterModel
    
    # clustering による値の算出
    def calc_cluster_probability(self):
        word_vectors = self.get_word2Vec()
        
        # 出力の格納
        idx = self.get_clusterModel().predict(word_vectors)
        # Get probabilities of cluster assignments.
        idx_proba = self.get_clusterModel().predict_proba(word_vectors)

        return (idx, idx_proba)
    
    # clustering の結果をセット
    def set_cluster(self):
        # clustering によって算出した値を使用する
        idx_cluster, idx_proba = self.calc_cluster_probability()
        
        for idx, word in enumerate(self.vocabulary):
            word.set_cluster_idx(idx_cluster[idx])
            word.set_cluster_probability(idx_proba[idx])
        return
    
    # tf-idf を計算い、idf値をセット
    # 入力
    # lst_lst_word 各文書ごとの単語リストのリスト
    def calc_idf_by_word(self, lst_lst_word):
        # 空白区切りの1つの文字列にする
        lst_context = [" ".join(lst_word) for lst_word in lst_lst_word]
        
        # tf-idfにかける
        tfv = TfidfVectorizer(strip_accents='unicode',dtype=np.float32)
        tfidfmatrix_traindata = tfv.fit_transform(lst_context)

        # 単語
        featurenames = tfv.get_feature_names()
        # idf 値の抽出
        idf = tfv._tfidf.idf_

        return featurenames, idf
    
    # 語彙中の各単語に idf値のセッティング
    # 入力
    # lst_word str型の単語リスト
    # lst_idf  lst_word と同じ並びで対応するidf値
    def set_idf(self, lst_word, lst_idf):
        # lst_word と lst_idf が同じ長さでなければエラーを返す
        assert len(lst_word)==len(lst_idf), "Input lengths are not equal. lst_word's length is {0}, but lst_idf's length is {1}".format(len(lst_word), len(lst_idf))
       
        # idf値はすべての単語に設定されるわけではないので、存在する単語にだけ格納する
        # もしvocabulary 中の単語にidf値が設定されなかった場合、その単語を vocabulary から削除する
        for word in self.vocabulary:
            try:
                # vocabulary の単語と一致するのが何番目か格納しておく
                idx = lst_word.index(word.name)
                # 対応するインデックスがわかったので値を格納する
                word.set_idf(idx)
            except:
                self.remove_vocabulary(word.name)
#         for word, idf in zip(lst_word,lst_idf):
#             # vocabulary の単語と一致するのが何番目か格納しておく
#             idx = self.get_vocabulary().index(word)
#             # 対応するインデックスがわかったので値を格納する
#             self.vocabulary[idx].set_idf(idf)
        
        return
    
    # 全単語に対して clustered_vector を算出
    def make_clustered_vector(self):
        for word in self.vocabulary:
#             # idf値, word vector が作られているもののみ作る
#             if word.get_idf() != 0 and word.get_vector().shape[0] != 0:
            word.set_clustered_vector(word.calc_clustered_vector())
        return
    
    # 文書のセッティング
    # 入力
    # lst_lst_word 各Documentごとに1つのリストに単語をまとめたもののリスト
    def set_documents(self, lst_lst_word):
        self.documents = []
        for idx, lst_word in enumerate(lst_lst_word):
            # 文書の単語に一致する Word型のリストを vocabulary から抽出する
            lst_word_class = [vocab for vocab in self.vocabulary if len(list(filter((lambda x: vocab.match(x)), lst_word)))>0]
            self.documents.append(Document(idx, lst_word_class))
        return
    
    # Documents の取得
    def get_documents(self):
        return self.documents
    
    # 文書からstr型リストに変換する
    def get_vocab_from_documants(self):
        # 出力の定義
        lst_word = []
        for document in self.documents:
            lst_word += document.words
        return lst_word
    
    # 全Documentに対し、所属するすべての単語のベクトルを平均化する
    def make_meanDocumentVector(self):
        for document in self.get_documents():
            # 各Document で平均ベクトルを作成してセット
            document.set_meanWordVector(document.calc_meanWordVector())
        return            
    
    # 全文書に対してスパース平均ベクトルを作成してセットする
    def make_sparceDocumentVector(self):
        for document in self.get_documents():
            document.set_sparceMeanVector(document.calc_sparceMeanVector(self.threshold))
        return
    
    # 文書idから sparce vector を出力する
#     def get_sparceDocumentVector(self, idx):
        
    
    
# Dpcument クラス
class Document():
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