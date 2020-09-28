import sys
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
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

    #@document_it
    #def test_allone_aveData(self):
    #    'configモジュールの関数(aveData)をテストする関数'
    #    '引数のデータフレーム型オブジェクトの要素は全て1'
    #    input_rows = 10
    #    input_columns = 3
    #    #関数aveDataを使用し、データフレーム型オブジェクトを返値
    #    df1 = config.aveData(make_allone_dataframe(row=input_rows, columns=input_columns))
    #    #上記df1と結果が等しくなるように、データフレーム型オブジェクトを作成
    #    df2 = pd.DataFrame(
    #            np.reshape(np.ones(1*input_columns),
    #                       (1, input_columns)
    #                      ),
    #            )
    #    #アサーション
    #    assert_frame_equal(df1, df2)

    #@document_it
    #def test_arange_aveData(self):
    #    'configモジュールの関数(aveData)をテストする関数'
    #    '引数のデータフレーム型オブジェクトの要素はnp.arangeで作成'
    #    input_rows = 10
    #    input_columns = 3
    #    #関数aveDataを使用し、データフレーム型オブジェクトを返値
    #    df1 = config.aveData(make_arange_dataframe(row=input_rows, columns=input_columns))
    #    #上記df1と結果が等しくなるように、データフレーム型オブジェクトを作成
    #    arr = np.reshape(np.arange(input_rows*input_columns), (input_rows, input_columns))
    #    df_tmp = (pd.DataFrame(arr)).mean(axis=0)
    #    df2 = pd.DataFrame(df_tmp, columns=[0]).T
    #    #アサーション
    #    assert_frame_equal(df1, df2)


    @document_it
    #@print_dataframe
    def test_random_aveData(self):
        'configモジュールの関数(aveData)をテストする関数'
        '引数のデータフレーム型オブジェクトの要素はnp.random.randで作成'
        'FIXME:config.AVERAGE=input_rowsでなければ、テストが通らない.→少なくともテストコードに問題あり.'
        input_rows = 10
        input_columns = 3
        df_rand = make_random_dataframe(row=input_rows, columns=input_columns)
        #関数aveDataを使用し、データフレーム型オブジェクトを返値
        df1 = config.aveData(df_rand)
        #上記df1と結果が等しくなるように、データフレーム型オブジェクトを作成
        df_tmp = df_rand.mean(axis=0)
        df2 = pd.DataFrame(df_tmp, columns=[0]).T
        print('df1:\n', df1)
        print('df2:\n', df2)
        #アサーション
        assert_frame_equal(df1, df2)


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
