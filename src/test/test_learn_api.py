import os
import glob
import numpy as np
import datetime
import unittest


class IterAdd():
    '''
    テスト用のイテレータプロトコル
    '''
    def __init__(self, input_date_time):
        self.date_time = input_date_time

    def __iter__(self):
        for _ in range(5):
            self.date_time = self.date_time + datetime.timedelta(microseconds=np.random.binomial(1, 0.6)*100000)
            yield self.date_time.strftime('%M:%S.%f')


class TestApi(unittest.TestCase):

    def setUp(self):
        '''
        イテレータプロトコルの簡易出力テスト
        '''
        date_time = datetime.datetime(2018, 12, 19, 14, minute=00, second=00, microsecond=0)
        iter_add = IterAdd(date_time)
        for i in iter_add:
            print(i)

    def tearDown(self):
        pass

    def _test_os_path_isfile(self):
        '''
        os.path.isfileの学習用テスト
        '''
        self.assertTrue(os.path.isfile('test_plot/*'))

    def _test_glob_glob(self):
        '''
        glob.globの学習用テスト

        Notes
        -----
        書き方その１: unittestでなく、標準のassert文を利用
        assert glob.glob('./test_plot/*.png') is not None  # 的を得ていない書き方。Noneはシングルトンオブジェクト。
        assert glob.glob('./test_plot/*.png')
        書き方その２: unittestのassertメソッド(assertTrue)を利用
        '''
        self.assertTrue(glob.glob('./test_plot/*.png'))


if __name__ == '__main__':
    unittest.main()
