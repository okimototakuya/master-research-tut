import sys
import numpy as np
import pandas as pd
#from pandas.util.testing import assert_frame_equal     # pandas.util.testing:FutureWarning → pandas.testing
import unittest
from pprint import pprint
sys.path.append('../')
from main import config


##################################################################
##########テストクラス############################################
##################################################################
class TestConfig(unittest.TestCase):
    'configモジュールの関数(aveData)をテストするクラス'
    def setUp(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    '関数make_allone_dataframeの詳細情報を出力'
    #テストの実行
    unittest. main()
