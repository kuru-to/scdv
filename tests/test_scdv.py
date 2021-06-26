#!/usr/bin/env python

"""Tests for `scdv` package."""
import unittest
import warnings
import itertools

from scdv.scdv import SCDV


class TestScdv(unittest.TestCase):
    """Tests for `scdv` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        warnings.simplefilter('ignore')
        self._lst_lst_word = [
            ["apple", "banana"], ["corch", "banana", "empty", ""], ["apple"]
        ]
        self._embedding_dimension = 100
        self._num_cluster = 3
        self._aSCDV = SCDV(
            embedding_dimension=self._embedding_dimension,
            num_cluster=self._num_cluster
        )
        self._aSCDV.lst_lst_word = self._lst_lst_word
        self._aSCDV.set_vocabulary()

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_set_vocabulary(self):
        """
        テスト項目:
            * 語彙の数は `_lst_lst_word` で唯一になる単語の数と同じか
        """
        test_voc = self._aSCDV.vocabulary
        num_vocab = len(set(itertools.chain.from_iterable(self._lst_lst_word)))
        self.assertEqual(len(test_voc), num_vocab)

    def test_word2Vec(self):
        """
        テスト項目:
            * 全単語に対してベクトルが定義されているか確認する
            * 埋め込み次元数は等しいか確認する
            * vocabulary 中の全単語に対してベクトルが定義されているか確認する.
                定義されていない単語は vocabulary から削除されているはずなのででない
            * モデルにおけるベクトルとWord class におけるベクトルは等しいか確認する
        """
        # word2vec 作成
        self._aSCDV.make_word2VecModel()
        # set
        self._aSCDV.set_word2Vec_to_vocab()
        vec_model = self._aSCDV.word2Vec_model

        for word in self._aSCDV.vocabulary:
            vec_model[str(word)]

        self.assertEqual(vec_model.wv.syn0.shape[1], self._embedding_dimension)

        for word in self._aSCDV.vocabulary.values():
            self.assertTrue((vec_model[str(word)] == word.vector).all)

        word_vectors = self._aSCDV.get_word2Matrix()

        self.assertEqual(word_vectors.shape[0], len(self._aSCDV.vocabulary))
        self.assertEqual(word_vectors.shape[1], self._embedding_dimension)

    def test_clustering(self):
        """
        テスト項目:
            * `calc_cluster_probability` の出力のうち確率の行列が, クラスタ数と一致するか
            * `Vocabulary` class の `Word` にクラスタの情報が格納されたか
        """
        # word2vec 作成
        self._aSCDV.make_word2VecModel()
        # set
        self._aSCDV.set_word2Vec_to_vocab()

        # clustering model の作成
        self._aSCDV.make_cluster_model()

        idx_cluster, idx_proba = self._aSCDV.calc_cluster_probability()

        self.assertEqual(idx_proba.shape[1], self._num_cluster)

        # clustering 結果のセット
        self._aSCDV.set_cluster()

        for idx, word in enumerate(self._aSCDV.vocabulary.values()):
            self.assertEqual(word.cluster_idx, idx_cluster[idx])
            self.assertTrue((word.cluster_probability == idx_proba).all)

    def test_idf(self):
        """
        テスト項目:
            * 各単語のidf値はvocaburalyに存在する単語か
            * idf値が設定されている vocabulary に対し、値は一致するか
        """
        # idf値算出
        feature_names, idf = self._aSCDV.calc_idf()

        for word in feature_names:
            self.assertTrue(word in self._aSCDV.vocabulary)

        self._aSCDV.set_idf(feature_names, idf)

        for word in self._aSCDV.vocabulary.values():
            idf_word = word.idf
            # idf値が設定されている場合に値のチェック
            if idf_word != 0:
                self.assertEqual(idf_word, idf[feature_names.index(str(word))])

    def test_set_documents(self):
        """
        テスト項目:
            * 文書のインデックスから得られる単語が語彙中の同じインデックスの単語と一致するか
        """
        self._aSCDV.set_documents()
        for idx, lst_word in enumerate(self._lst_lst_word):
            test_doc = self._aSCDV.documents[idx]
            test_words = [str(w) for w in test_doc.lst_word]
            self.assertEqual(set(lst_word), set(test_words))

    def test_train(self):
        """
        テスト項目:
            * SCDVの次元が 単語ベクトルの次元数 * クラスタ数と一致するか確認
        """
        self._aSCDV.train(self._lst_lst_word)

        # 各ベクトルの次元が一致するか確認
        for idx in range(len(self._aSCDV.documents)):
            self._aSCDV.get_sparce_document_vector(idx)


if __name__ == "__main__":
    unittest.main()
