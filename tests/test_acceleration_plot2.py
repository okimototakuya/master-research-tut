import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import unittest
sys.path.append('../main')
from acceleration_plot2 import DataframePlotter
from acceleration_plot2 import TimePredDataframePlotter
from acceleration_plot2 import Acc1Acc2DataframePlotter
import config


class TestAcceleration_plot2(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_plot_TimePredDataframePlotter(self):
        tpdfp = TimePredDataframePlotter(config.dataframe, 30)
        print(config.dataframe)
        tpdfp.plot(tuple(config.acc))
        assert glob.glob('./test_plot/*.png') is not None

    def test_plot_Acc1Acc2DataframePlotter(self):
        aadfp = Acc1Acc2DataframePlotter(config.dataframe, 30)
        #print(aadfp.df)
        aadfp.plot()
        assert glob.glob('./test_plot/*.png') is not None


if __name__ == '__main__':
    '本番プログラム：configモジュールのdataframe変数にプロットするDataFrame型変数を格納する.'
    config.dataframe = pd.DataFrame(
                      np.reshape(np.arange(30), (10,3)),
                      columns = ['Acceleration_x', 'Acceleration_y', 'Acceleration_z'],
                     )
    config.pred = np.ones(10)
    unittest.main()
