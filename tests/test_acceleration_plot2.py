import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import unittest
sys.path.append('../main')
from acceleration_plot2 import DataframeMaker
from acceleration_plot2 import DataframePlotter
from acceleration_plot2 import TimePredDataframePlotter
from acceleration_plot2 import Acc1Acc2DataframePlotter
import config


class TestAcceleration_plot2(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _test_plot_TimePredDataframePlotter(self):
        'TimePredDataframePlotterクラスのplot関数をテスト.'
        '指定ディレクトリ下にpngファイルが生成されたかどうかでアサート.'
        '正しくプロットされているかどうかはナイーブに確認.'
        tpdfp = TimePredDataframePlotter(config.data_sampled_by_func, 30, 's')
        #tpdfp = TimePredDataframePlotter(config.data_sampled_by_func, 30, 'p')
        #print(config.data_sampled_by_func)
        #tpdfp.state = 'p'
        tpdfp.save_graph_to_path = './test_plot1/'  # 互いに独立したテストにするため
        tpdfp.plot()    # 複数のグラフを生成
        #assert glob.glob('./test_plot/*.png') is not None
        self.assertTrue(glob.glob('./test_plot1/*.png'))

    def _test_plot_Acc1Acc2DataframePlotter(self):
        '上記テストメソッドと同じ.'
        aadfp = Acc1Acc2DataframePlotter(config.data_sampled_by_func, 30, 's')
        #aadfp = Acc1Acc2DataframePlotter(config.data_sampled_by_func, 30, 'p')
        #print(aadfp.df)
        aadfp.save_graph_to_path = os.path.join('./test_plot2/', 'demo.png')  # 互いに独立したテストにするため
        aadfp.plot()    # １つのグラフを生成
        #assert glob.glob('./test_plot/*.png') is not None
        self.assertTrue(glob.glob('./test_plot2/*.png'))

    def test_sample_data(self):
        # 開始オフセットに０を指定するとバグる:指定ディレクトリにファイル自体は生成されるが、中身は空っぽ
        DataframeMaker.sample_data('./test_sample/demo', config.data_read_by_api, 0, 6) 
        self.assertTrue('glob.glob(./test_sample/demo)')

    def _test_connect_dataframe(cls):
        predict = pd.DataFrame(config.pred_by_prob_model, columns=['pred'])
        pd.concat(axis=1)


if __name__ == '__main__':
    '本番プログラム：configモジュールのdataframe変数にプロットするDataFrame型変数を格納する.'
    'デモ：小規模のDataFrame型変数を生成し、正しくプロットできているかテストする.'
    # デモDataFrame型変数の生成
    config.data_sampled_by_func = pd.DataFrame(
                      #np.reshape(np.arange(30), (10,3)),
                      (np.arange(30)).reshape(10,3),
                      columns = ['Acceleration_x', 'Acceleration_y', 'Acceleration_z'],
                     )
    # デモ予測値の生成
    config.pred_by_prob_model = np.ones(10)
    unittest.main()
