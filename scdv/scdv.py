#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Main module.
"""
from typing import List, Union
import copy

from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture

from scdv.word import Word
from scdv.vocabulary import Vocabulary
from scdv.document import Document


class SCDV:
    """ Doc2Vec の1手法である Sparce Composite Document Vectors の実装

    Attributes:
        lst_lst_word: 文書ごとの単語のリスト, のリスト
        num_cluster: クラスタ数
        random_seed: 乱数の種
        threshold: sparce な vector を作成する際に0にする閾値
        embedding_dimension: Word vector dimensionality
        max_iter: Gausian Mixture Model 作成時の反復回数最大値
    """
    def __init__(
        self,
        num_cluster: int = 60,
        random_seed: int = 0,
        threshold: float = 0.01,
        embedding_dimension: int = 200,
        max_iter: int = 50
    ):
        self._num_cluster = num_cluster
        self._random_seed = random_seed
        self._threshold = threshold
        self._embedding_dimension = embedding_dimension
        self._max_iter = max_iter

    @property
    def lst_lst_word(self):
        return copy.deepcopy(self._lst_lst_word)

    @lst_lst_word.setter
    def lst_lst_word(self, lst_lst_word: List[List[str]]):
        """
        Attributes:
            lst_lst_word: 文書ごとの単語のリスト, のリスト
        """
        self._lst_lst_word = copy.deepcopy(lst_lst_word)

    def set_vocabulary(self):
        """ Setter of vocabulary

        Attributes:
            vocabulary (:obj:`Vocabulary`): 登録された語彙の一覧
        """
        self._aVocabulary = Vocabulary(self.lst_lst_word)

    @property
    def vocabulary(self):
        return self._aVocabulary

    def remove_vocabulary(self, word: Union[str, Word]):
        """ 語彙から削除する

        Args:
            word: 削除したい単語
        """
        del self.vocabulary[str(word)]

    def make_word2VecModel(
        self,
        min_word_count: int = 0,
        num_workers: int = 1,
        context: int = 1,
        downsampling: float = 1e-3,
    ):
        """Make word2vec

        Args:
            min_word_count (optional): Minimum word count
            num_workers (optional): Number of threads to run in parallel
            context (optional): Context window size
            downsampling (optional): Downsample setting for frequent words

        Attributes:
           word2vec (:obj:`Word2Vec`): lst_lst_word から作成した Word2Vec

        Note:
            word2vec のパラメータについては引数で対応. 足りなければ順次追加
        """
        print("Training word2Vec model...")
        self.word2vec = Word2Vec(
            self.lst_lst_word,
            workers=num_workers,
            hs=0,
            sg=1,
            negative=10,
            iter=25,
            size=self._embedding_dimension,
            min_count=min_word_count,
            window=context,
            sample=downsampling,
            seed=self._random_seed
        )

        # L2ノルムで正規化.メモリが足りない場合は replace=True
        self.word2vec.init_sims(replace=True)

        # Get wordvectors for all words in vocabulary.

    @property
    def word2Vec_model(self):
        """word2vec のモデル取得

        Note:
            `make_word2VecModel` method を事前に実行していない場合, エラーを返す
        """
        try:
            word2VecModel = self.word2vec
        except AttributeError:
            assert False, "word2vec model has not been made yet."
        return word2VecModel

    def set_word2Vec_to_vocab(self):
        """word2vec のモデルからベクトルを抽出して語彙中の各単語に格納

        Note:
            * word vector が作成されていれば格納
            * word vector が作成されていない単語であれば、vocabulary から remove
        """
        for word in self.vocabulary.values():
            try:
                word.vector = self.word2Vec_model[str(word)]
            except KeyError:
                self.remove_vocabulary(str(word))

    def get_word2Matrix(self):
        """語彙から全単語ベクトルを concat した行列を出力する

        Returns:
            np.array: 行に単語ベクトルを持つ行列
        """
        word_vectors = np.concatenate([
                np.array(word.vector).reshape([1, self._embedding_dimension])
                for word in self.vocabulary.values()
            ], axis=0
        )
        return word_vectors

    def make_cluster_model(self):
        """クラスタリングのモデル作成"""
        # Initalize a GMM object and use it for clustering.
        clf = GaussianMixture(
            n_components=self._num_cluster,
            covariance_type="tied",
            init_params='kmeans',
            max_iter=self._max_iter
        )

        # Get cluster assignments.
        print("Training clustering model...")
        clf.fit(self.get_word2Matrix())

        # モデルの格納
        self._cluster_model = clf

    # clustering のモデルを取得する
    def get_clusterModel(self):
        try:
            clusterModel = self._cluster_model
        except AttributeError:
            assert False, "clustering model has not been made yet."
        return clusterModel

    def calc_cluster_probability(self):
        """cluster への所属確率の出力

        Returns:
            List[int]: 所属確率最大のクラスタのインデックス
            List[np.array]: 各単語ごとの各クラスタへの所属確率
        """
        word_vectors = self.get_word2Matrix()

        # 出力の格納
        idx = self.get_clusterModel().predict(word_vectors)
        # Get probabilities of cluster assignments.
        idx_proba = self.get_clusterModel().predict_proba(word_vectors)
        return (idx, idx_proba)

    def set_cluster(self):
        """clustering の結果を各単語にセット"""
        idx_cluster, idx_proba = self.calc_cluster_probability()

        for idx, word in enumerate(self.vocabulary.values()):
            word.cluster_idx = idx_cluster[idx]
            word.cluster_probability = idx_proba[idx]

    def calc_idf(self):
        """idf値を計算"""
        # 空白区切りの1つの文字列にする
        lst_context = [
            " ".join([word for word in lst_word])
            for lst_word in self.lst_lst_word
        ]

        # tf-idfにかける
        tfv = TfidfVectorizer(strip_accents='unicode', dtype=np.float32)
        _ = tfv.fit_transform(lst_context)

        # 単語
        featurenames = tfv.get_feature_names()
        # idf 値の抽出
        idf = tfv._tfidf.idf_

        return featurenames, idf

    def set_idf(self, lst_word: List[str], lst_idf: List[float]):
        """語彙中の各単語に idf値のセッティング

        Args:
            lst_word: idf値が算出された, str型の単語リスト
            lst_idf: lst_word と同じ並びで対応するidf値

        Note:
            * lst_word と lst_idf が同じ長さでなければエラーを返す
            * idf値はすべての単語に設定されるわけではないので, 存在する単語にだけ格納する
            * もしvocabulary 中の単語にidf値が設定されなかった場合, その単語を vocabulary から削除する
        """
        num_word = len(lst_word)
        num_idf = len(lst_idf)
        message = "Input lengths must be equal. 1st is {0}, but 2nd is {1}"
        assert num_word == num_idf, message.format(num_word, num_idf)

        # for文の中で語彙を削除するとfor文が回らなくなってしまうので,
        # 削除する単語は別だし
        lst_remove = []
        for w in self.vocabulary.values():
            try:
                idx = lst_word.index(str(w))
                w.idf = lst_idf[idx]
            except ValueError:
                lst_remove.append(w)

        for removed_word in lst_remove:
            del self.vocabulary[removed_word]

    def set_documents(self):
        """文書のセッティング

        Note:
            * `vocabulary` 中の `Word` class を文書に格納する
        """
        self._documents = []
        for idx, lst_word in enumerate(self.lst_lst_word):
            lst_word_in_doc = [
                self.vocabulary[word.lower()] for word in lst_word
                if word.lower() in self.vocabulary.keys()
            ]
            self._documents.append(Document(idx, lst_word_in_doc))

    @property
    def documents(self):
        return self._documents

    def calc_sparce_mean_vector(self, doc: Document):
        """文書の sparce mean vector の計算"""
        mean_vector = doc.mean_word_vector
        return np.where(np.abs(mean_vector) < self._threshold, 0, mean_vector)

    def set_sparce_mean_vector(self):
        """各文書に sparce mean vector を set"""
        for doc in self.documents:
            doc.sparce_mean_vector = self.calc_sparce_mean_vector(doc)

    def train(self, lst_lst_word: List[List[str]]):
        """ SCDV 実行して文書ベクトルの作成

        Note:
            * 文書ベクトルは `Document` クラスをセットしてしまえば計算可能
        """
        self.lst_lst_word = lst_lst_word
        self.set_vocabulary()

        # word2vec 作成
        self.make_word2VecModel()
        self.set_word2Vec_to_vocab()

        # cluster 作成
        self.make_cluster_model()
        self.set_cluster()

        # idf 値算出
        feature_names, idf = self.calc_idf()
        self.set_idf(feature_names, idf)

        # Document セット
        self.set_documents()

        # mean word vector の計算は documents の内部で行われる
        # sparce mean vector set
        self.set_sparce_mean_vector()

    def get_sparce_document_vector(self, idx: int):
        """文書idから sparce vector を出力する"""
        return self.documents[idx].sparce_mean_vector
