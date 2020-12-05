#!/usr/bin/env python

"""Tests for `scdv` package."""

import unittest
from click.testing import CliRunner
import warnings

from scdv.scdv import SCDV
from scdv import cli

class TestScdv(unittest.TestCase):
    """Tests for `scdv` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        warnings.simplefilter('ignore')

    def tearDown(self):
        """Tear down test fixtures, if any."""

    # SCDVのコンストラクタのテスト
    def test_SCDV_constract(self):
        num_cluster=10
        random_seed = 1
        threshold = 0.1
        embedding_dimension = 300
        
        model = SCDV(num_cluster, random_seed, threshold, embedding_dimension)
        
        self.assertEqual(model.num_cluster, num_cluster)
        self.assertEqual(model.random_seed, random_seed)
        self.assertEqual(model.threshold, threshold)
        self.assertEqual(model.embedding_dimension, embedding_dimension)
        return

    # remove_vocabulary のテスト
    def test_SCDV_remove_vocabulary(self):
        lst_lst_word = [["a", "b"], ["c", "d", "e", ""], ["a"]]
        remove_word = "a"
        
        model = SCDV()
        model.set_vocabulary(lst_lst_word)
        model.remove_vocabulary(remove_word)
        
        lst_answer = []
        for lst_word in lst_lst_word:
            lst_answer += lst_word
        lst_answer = list(set(lst_answer))
        lst_answer.remove(remove_word)
        
        # 削除したい単語が消えているか確認
        self.assertTrue(remove_word not in model.get_vocabulary())
        
        # 削除したくない単語が消えていないか確認
        for ans in lst_answer:
            self.assertTrue(ans in model.get_vocabulary())
        return

    # set_vocablary のテスト
    def test_SCDV_set_vocabulary(self):
        lst_lst_word = [["a", "b"], ["c", "d", "e", ""], ["a"]]
        lst_word = []
        for lst_input in lst_lst_word:
            lst_word+=lst_input
        
        model = SCDV()
        model.set_vocabulary(lst_lst_word)
        
        # 語彙の多さは同じか
        self.assertEqual(len(model.vocabulary), len(set(lst_word)))
        
        # 各単語は語彙に登録されているか
        for word in lst_word:
            self.assertTrue(word in [word_vocab.name for word_vocab in model.vocabulary])
        return
        
    # make_word2VecModel のテスト
    def test_SCDV_make_word2VecModel(self):
        lst_lst_word = [["a", "b"], ["c", "d", "e", ""], ["a"]]
        embedding_dimension = 100
        
        model = SCDV(embedding_dimension=embedding_dimension)
        model.set_vocabulary(lst_lst_word)
        
        # word2vec 作成
        model.make_word2VecModel(lst_lst_word)
        
        vec_model = model.get_word2VecModel()
        
        # 全単語に対してベクトルが定義されているか確認する
        for word in model.get_vocabulary():
            try:
                vec_model[word]
            except NameError:
                self.assertTrue(False)
            
        # 埋め込み次元数は等しいか確認する
        self.assertEqual(vec_model.wv.syn0.shape[1], embedding_dimension)
        return

    # set_word2VecModel のテスト
    def test_SCDV_set_word2VecModel(self):
        lst_lst_word = [["a", "b"], ["c", "d", "e", ""], ["a"]]
        
        model = SCDV()
        model.set_vocabulary(lst_lst_word)
        
        # word2vec 作成
        model.make_word2VecModel(lst_lst_word, min_word_count=2)
        # set
        model.set_word2Vec()
        
        vec_model = model.word2vec
        
        # vocabulary 中の全単語に対してベクトルが定義されているか確認する
        # 定義されていない単語は vocabulary から削除されているはずなのででない
        for word in model.vocabulary:
            try:
                word.get_vector()
                
                # モデルにおけるベクトルとWord class におけるベクトルは等しいか確認する
                self.assertTrue((vec_model[word.get_name()] == word.get_vector()).all)
            except NameError:
                self.assertTrue(False)
        return

    # get_word2vec のテスト
    def test_SCDV_get_word2Vec(self):
        lst_lst_word = [["a", "b"], ["c", "d", "e", ""], ["a"]]
        embedding_dimension = 100
        
        model = SCDV(embedding_dimension=embedding_dimension)
        model.set_vocabulary(lst_lst_word)
        
        # word2vec 作成
        model.make_word2VecModel(lst_lst_word)
        # set
        model.set_word2Vec()
        # get
        word_vectors = model.get_word2Vec()

        self.assertEqual(word_vectors.shape[0], len(model.get_vocabulary()))
        self.assertEqual(word_vectors.shape[1], embedding_dimension)
        return

    # calc_cluster_probability のテスト
    def test_SCDV_calc_cluster_probability(self):
        lst_lst_word = [["a", "b"], ["c", "d", "e", ""], ["a"]]
        num_cluster = 3
        
        model = SCDV(num_cluster=num_cluster)
        model.set_vocabulary(lst_lst_word)
        
        # word2vec 作成
        model.make_word2VecModel(lst_lst_word)
        # set
        model.set_word2Vec()
        
        # clustering model の作成
        model.make_clusterModel()
        
        idx, idx_proba = model.calc_cluster_probability()
        
        self.assertEqual(idx_proba.shape[1], num_cluster)
        self.assertEqual(sum([1 if idx_cluster >= num_cluster else 0 for idx_cluster in idx]), 0)
        return

    # set_cluster のテスト
    def test_SCDV_set_cluster(self):
        lst_lst_word = [["a", "b"], ["c", "d", "e", ""], ["a"]]
        num_cluster = 3
        
        model = SCDV(num_cluster=num_cluster)
        model.set_vocabulary(lst_lst_word)
        
        # word2vec 作成
        model.make_word2VecModel(lst_lst_word)
        # set
        model.set_word2Vec()
        
        # clustering model の作成
        model.make_clusterModel()
        model.set_cluster()
        idx_cluster, idx_proba = model.calc_cluster_probability()
        
        for idx, word in enumerate(model.vocabulary):
            self.assertEqual(word.get_cluster_idx(), idx_cluster[idx])
            self.assertTrue((word.get_cluster_probability==idx_proba).all)
        return

    # calc_idf_by_word のテスト
    def test_SCDV_calc_idf_by_word(self):
        lst_lst_word = [["apple", "banana"], ["corch", "banana", "empty", ""], ["apple"]]
        
        model = SCDV()
        model.set_vocabulary(lst_lst_word)
        
        # idf値算出
        feature_names, _ = model.calc_idf_by_word(lst_lst_word)
        
        # 各単語のidf値はvocaburalyに存在する単語か
        for word in feature_names:
            self.assertTrue(word in model.get_vocabulary())
        return

    # set_idf のテスト
    def test_SCDV_set_idf(self):
        lst_lst_word = [["apple", "banana"], ["corch", "banana", "empty", ""], ["apple", "I"]]
        
        model = SCDV()
        model.set_vocabulary(lst_lst_word)
        
        # idf 値算出
        feature_names, idf = model.calc_idf_by_word(lst_lst_word)
        model.set_idf(feature_names, idf)
        
        # idf値が設定されている vocabulary に対し、値は一致するか
        for word in model.vocabulary:
            idf_word = word.get_idf()
            # idf値が設定されている場合に値のチェック
            if idf_word != 0:
                self.assertTrue((idf_word == idf[feature_names.index(word.get_name())]).all)
        return

    # make_clustered_vector のテスト
    def test_SCDV_make_clustered_vector(self):
        lst_lst_word = [["apple", "banana"], ["corch", "banana", "empty", ""], ["apple"]]
        num_cluster = 3
        
        model = SCDV(num_cluster = num_cluster)
        model.set_vocabulary(lst_lst_word)
        
        # word2vec 作成
        model.make_word2VecModel(lst_lst_word)
        model.set_word2Vec()
        
        # clustering model の作成
        model.make_clusterModel()
        model.set_cluster()
        
        # idf 値算出・セット
        feature_names, idf = model.calc_idf_by_word(lst_lst_word)
        model.set_idf(feature_names, idf)
        
        # clustered_vector の算出
        model.make_clustered_vector()
        return

    # set_documentのテスト
    def test_SCDV_set_document(self):
        lst_lst_word = [["apple", "banana"], ["corch", "banana", "empty", ""], ["apple"]]
        remove_word= "banana"
        
        model = SCDV()
        model.set_vocabulary(lst_lst_word)
        model.remove_vocabulary(remove_word)
        model.set_documents(lst_lst_word)
        
        # document の数は一致するか
        self.assertEqual(len(lst_lst_word), len(model.documents))
        # 各単語は一致するか
        for idx, lst_word in enumerate(lst_lst_word):
            document = model.documents[idx]
            # 削除対象の単語を抜く
            if remove_word in lst_word:
                lst_word.remove(remove_word)
            self.assertEqual(len(document.words), len(lst_word))
            
            for word in lst_word:
                self.assertTrue(document.isExist_word(word))
        return

    # make_meanDocumentVector のテスト
    def test_SCDV_make_meanDocumentVector(self):
        lst_lst_word = [["apple", "banana"], ["corch", "banana", "empty", ""], ["apple"]]
        num_cluster = 3
        embedding_dimension = 100
        
        model = SCDV(num_cluster=num_cluster,embedding_dimension=embedding_dimension)
        model.set_vocabulary(lst_lst_word)
        
        # word2vec 作成
        model.make_word2VecModel(lst_lst_word)
        model.set_word2Vec()
        
        # cluster 作成
        model.make_clusterModel()
        model.set_cluster()
        
        # idf 値算出
        feature_names, idf = model.calc_idf_by_word(lst_lst_word)
        model.set_idf(feature_names, idf)
        
        # clustered_vector の算出
        model.make_clustered_vector()
        
        # Document セット
        model.set_documents(lst_lst_word)
        
        # 平均ベクトルセット
        model.make_meanDocumentVector()
        
        # 各ベクトルの次元が一致するか確認
        for document in model.get_documents():
            self.assertEqual(document.get_meanWordVector().shape[0], num_cluster*embedding_dimension)
            try:
                document.get_meanWordVector().shape[1]
                self.assertTrue(False)
            except:
                continue
        return

    # make_sparceDocumentVector のテスト
    def test_SCDV_make_sparceDocumentVector(self):
        lst_lst_word = [["apple", "banana"], ["corch", "banana", "empty", ""], ["apple"]]
        num_cluster = 3
        embedding_dimension = 100
        
        model = SCDV(num_cluster=num_cluster,embedding_dimension=embedding_dimension)
        model.set_vocabulary(lst_lst_word)
        
        # word2vec 作成
        model.make_word2VecModel(lst_lst_word)
        model.set_word2Vec()
        
        # cluster 作成
        model.make_clusterModel()
        model.set_cluster()
        
        # idf 値算出
        feature_names, idf = model.calc_idf_by_word(lst_lst_word)
        model.set_idf(feature_names, idf)
        
        # clustered_vector の算出
        model.make_clustered_vector()
        
        # Document セット
        model.set_documents(lst_lst_word)
        
        # 平均ベクトルセット
        model.make_meanDocumentVector()
        
        # sparce vector set
        model.make_sparceDocumentVector()
        
        # 各ベクトルの次元が一致するか確認
        for document in model.get_documents():
            self.assertEqual(document.get_sparceMeanVector().shape[0], num_cluster*embedding_dimension)
            try:
                document.get_meanWordVector().shape[1]
                self.assertTrue(False)
            except:
                continue
        return
        

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'scdv.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

if __name__ == "__main__":
    unittest.main()