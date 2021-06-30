import os
import glob
import unittest


class TestApi(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_os_path_isfile(self):
        'os.path.isfileの学習用テスト'
        self.assertTrue(os.path.isfile('test_plot/*'))

    def test_glob_glob(self):
        'glob.globの学習用テスト'
        # 書き方その１: unittestでなく、標準のassert文を利用
        #assert glob.glob('./test_plot/*.png') is not None  # 的を得ていない書き方。Noneはシングルトンオブジェクト。
        #assert glob.glob('./test_plot/*.png')
        # 書き方その２: unittestのassertメソッド(assertTrue)を利用
        self.assertTrue(glob.glob('./test_plot/*.png'))


if __name__ == '__main__':
    unittest.main()
