"""Main module."""
## create word2vec
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.mixture import GaussianMixture

from scdv.word import Word
from scdv.document import Document

class SCDV():    
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
    
    # lst_lst_word 語彙となる単語のリストのリスト（文書ごと）
    def set_vocabulary(self, lst_lst_word):
        # 展開
        lst_word = []
        for lst_input in lst_lst_word:
            lst_word += lst_input
        
        # 同じ単語が登録されないようにset型に変換してから登録する
        self.vocabulary = [Word(word) for word in set(lst_word)]
        return
    
    # str型のリストで出力する
    def get_vocabulary(self):
        return [word.get_name() for word in self.vocabulary].copy()
    
    # word 削除する単語. str型
    def remove_vocabulary(self, word):
        self.vocabulary = list(filter(lambda x: not x.match(word), self.vocabulary))
        return
        
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
                self.get_word2VecModel()[word.name]
                word.set_vector(self.get_word2VecModel()[word.name])
            # word vector が作成されていない単語であれば、vocabulary から remove
            except:
                self.remove_vocabulary(word.name)
        return
    
    # vocabulary からベクトルを出力する
    def get_word2Vec(self):
        word_vectors_tmp = np.array(self.vocabulary[0].get_vector())
        word_vectors = word_vectors_tmp.reshape([1,word_vectors_tmp.shape[0]])
        for word in self.vocabulary[1:]:
            word_vectors = np.concatenate([word_vectors, word.get_vector().reshape([1,word_vectors.shape[1]])], axis=0)
        return word_vectors
    
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
    
    def get_clusterModel(self):
        try:
            clusterModel = self.cluster_model
        except:
            assert False, "clustering model has not been made yet."
        return clusterModel
    
    def calc_cluster_probability(self):
        word_vectors = self.get_word2Vec()
        
        # 出力の格納
        idx = self.get_clusterModel().predict(word_vectors)
        # Get probabilities of cluster assignments.
        idx_proba = self.get_clusterModel().predict_proba(word_vectors)

        return (idx, idx_proba)
    
    def set_cluster(self):
        # clustering によって算出した値を使用する
        idx_cluster, idx_proba = self.calc_cluster_probability()
        
        for idx, word in enumerate(self.vocabulary):
            word.set_cluster_idx(idx_cluster[idx])
            word.set_cluster_probability(idx_proba[idx])
        return
    
    # tf-idf を計算し、idf値をセット
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
        
        return
    
    # 全単語に対して clustered_vector を算出
    def make_clustered_vector(self):
        for word in self.vocabulary:
            word.set_clustered_vector(word.calc_clustered_vector())
        return
    
    # lst_lst_word 各Documentごとに1つのリストに単語をまとめたもののリスト
    def set_documents(self, lst_lst_word):
        self.documents = []
        for idx, lst_word in enumerate(lst_lst_word):
            # 文書の単語に一致する Word型のリストを vocabulary から抽出する
            lst_word_class = [vocab for vocab in self.vocabulary if len(list(filter((lambda x: vocab.match(x)), lst_word)))>0]
            self.documents.append(Document(idx, lst_word_class))
        return
    
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
    
    # SCDV 実行して文書ベクトルの作成
    def run(self, lst_lst_word):
        self.set_vocabulary(lst_lst_word)

        # word2vec 作成
        self.make_word2VecModel(lst_lst_word)
        self.set_word2Vec()

        # cluster 作成
        self.make_clusterModel()
        self.set_cluster()

        # idf 値算出
        feature_names, idf = self.calc_idf_by_word(lst_lst_word)
        self.set_idf(feature_names, idf)

        # clustered_vector の算出
        self.make_clustered_vector()

        # Document セット
        self.set_documents(lst_lst_word)

        # 平均ベクトルセット
        self.make_meanDocumentVector()

        # sparce vector set
        self.make_sparceDocumentVector()
        return 

    # 文書idから sparce vector を出力する
    def get_sparceDocumentVector(self, idx):
        return self.documents[idx].get_sparceMeanVector()