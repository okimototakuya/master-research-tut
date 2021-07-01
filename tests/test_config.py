import sys
import numpy as np
import pandas as pd
#from pandas.util.testing import assert_frame_equal     # pandas.util.testing:FutureWarning → pandas.testing
import unittest
from pprint import pprint
sys.path.append('../')
from main import config


##################################################################
##########デコレータ##############################################
##################################################################
def document_it(func):
    '関数クラスを引数にとり、その関数の詳細パラメータを関数クラスにして返す'
    def new_function(*args, **kwargs):
        print('Running function:', func.__name__)
        print('Positional arguments:', args)
        print('Keyword arguments:', kwargs)
        result = func(*args, **kwargs)
        print('Result:\n', result)
        print('\n')
        return result

    return new_function


## FIXME:関数内のローカル変数を参照する特殊メソッドが分からない.
def print_dataframe(func):
    '関数内のローカル変数を表示する'
    def new_function(*args, **kwargs):
        print('関数内ローカル変数:', func().__bool__())
        result = func(*args, **kwargs)
        return result

    return new_function


##################################################################
##########テストに用いる諸関数####################################
##################################################################
def make_allone_dataframe(**kwargs):
    '行数と列数を引数にとり、データフレーム型オブジェクト(要素は全て１)を返す'
    df = pd.DataFrame(
            np.reshape(np.ones(kwargs['row']*kwargs['columns']),
                       (kwargs['row'], kwargs['columns'])
                      ),
            )
    #print(df)
    return df


def make_arange_dataframe(**kwargs):
    '行数と列数を引数にとり、データフレーム型オブジェクト(要素は全て１)を返す'
    df = pd.DataFrame(
            np.reshape(
                      np.arange(kwargs['row']*kwargs['columns']),
                      (kwargs['row'], kwargs['columns'])
                      ),
            )
    #print(df)
    return df


def make_random_dataframe(**kwargs):
    '行数と列数を引数にとり、データフレーム型オブジェクト(要素は全て１)を返す'
    df = pd.DataFrame(
            np.reshape(
                      np.random.rand(kwargs['row']*kwargs['columns']),
                      (kwargs['row'], kwargs['columns'])
                      ),
            )
    #print(df)
    return df


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
    #デコレータ使用時
    #make_allone_dataframe()

    #デコレータを使用せず、document_itオブジェクトを明示的に使用時
    #cooler_make_allone_dataframe = document_it(make_allone_dataframe)
    #cooler_make_allone_dataframe(row=10, columns=3)
    #cooler_make_allone_dataframe2 = document_it(make_allone_dataframe)
    #cooler_make_allone_dataframe2(row=1, columns=3)

    #テストの実行
    unittest. main()
