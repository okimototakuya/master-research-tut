import os
import sys
import glob
import subprocess
import datetime
import unittest
import numpy as np
import pandas as pd
sys.path.append('./src/main')
import acceleration_plot3 as ap3


AMOUNT_OF_ROW = 30  # テストcsvファイルの列数


class IterAddMicrosecond():
    'time特徴量について、1マイクロ秒加算'

    def __init__(self, input_date_time):
        self.date_time = input_date_time

    def __iter__(self):
        'イテレータプロトコル'
        p = 0.6 # 1マイクロ秒足す確率
        for _ in range(AMOUNT_OF_ROW):
            self.date_time = self.date_time + datetime.timedelta(microseconds=np.random.binomial(1, p)*100000) # ベルヌーイ分布に従って1マイクロ秒加算
            #self.date_time = self.date_time + datetime.timedelta(microseconds=1*100000) # 常に1マイクロ秒加算
            yield self.date_time.strftime('%M:%S.%f')


# HACK:2020.12.7:テストDataFrame型変数df_real_columnsの要素について、密度関数の出力値を固定できないか
# → 「平均値の計算」をテストする際、テスト値をベタ書きでソースコードに書いておけるので便利
# FIXME: 2021.12.5: プロダクトコードの方で、関数average_dataにpd.DataFrameを与える前に、列'time'をstring型 → pd.datetime64[ns]型にキャストしている。
df_real_columns = pd.DataFrame(
    {    # テストDataFrame型変数
        'Unnamed: 0':range(AMOUNT_OF_ROW),
        'line':range(AMOUNT_OF_ROW),
        'time':[ms for ms in IterAddMicrosecond(datetime.datetime(2018, 12, 19, 14, minute=00, second=00, microsecond=0))],
        'Acceleration(X)[g]':np.random.rand(AMOUNT_OF_ROW)*10-5,   # 一様分布に従って値を出力
        'Acceleration(Y)[g]':np.random.rand(AMOUNT_OF_ROW)*10-5,   # 一様分布に従って値を出力
        'Acceleration(Z)[g]':np.random.rand(AMOUNT_OF_ROW)*10-5,   # 一様分布に従って値を出力
        'AngularRate(X)[dps]':np.random.rand(AMOUNT_OF_ROW)*600-300,   # 一様分布に従って値を出力
        'AngularRate(Y)[dps]':np.random.rand(AMOUNT_OF_ROW)*600-300,   # 一様分布に従って値を出力
        'AngularRate(Z)[dps]':np.random.rand(AMOUNT_OF_ROW)*600-300,   # 一様分布に従って値を出力
        #'Acceleration(X)[g]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        #'Acceleration(Y)[g]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        #'Acceleration(Z)[g]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        #'AngularRate(X)[dps]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        #'AngularRate(Y)[dps]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        #'AngularRate(Z)[dps]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        #'Temperature[degree]':np.random.randn(AMOUNT_OF_ROW)+18,   # 正規分布に従って値を出力
        'Pressure[hPa]':np.random.randn(AMOUNT_OF_ROW)+1017,   # 正規分布に従って値を出力
        'Temperature[degree]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        'Pressure[hPa]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        'MagnetCount':np.zeros(AMOUNT_OF_ROW, dtype=int), # 常に値0を出力
        'MagnetSwitch':np.zeros(AMOUNT_OF_ROW, dtype=int), # 常に値0を出力
        #'onCrossroad':0, # 常に値0を出力
        #'crossroadID':0, # 常に値0を出力
        'onCrossroad':np.random.binomial(1, 0.01, AMOUNT_OF_ROW), #ベルヌーイ分布に従ってブール値を出力
        'crossroadID':np.random.binomial(1, 0.01, AMOUNT_OF_ROW), #ベルヌーイ分布に従ってブール値を出力
    },
    )


class TestAverageData(unittest.TestCase):
    '''
    関数ap3.average_dataについてテスト
    '''
    def setUp(self):
        '''
        テストcsvファイルを書込
        '''
        df_real_columns.to_csv('./test_dataset/demo.csv')
        #subprocess.call(['sed', '\'1', 's/,//\'', './test_dataset/demo.csv', '>', './test_dataset/demo.csv'])
        #subprocess.getoutput('sed -i -e \'1 s/,//\' ./test_dataset/demo.csv')   # 書き出したテストcsvファイルの先頭行頭のカンマを削除

    def tearDown(self):
        '''
        次回のテストのためにテストcsvファイルを削除
        '''
        os.remove('./test_dataset/demo.csv')

    #def _test_average_data_in_all_section_and_return_series_older(self):
    #    '''
    #    各columnsについて、全区間を算術平均し、計算結果をpd.Series型オブジェクトで返したかテスト
    #    '''
    #    # 1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
    #    df_test = ap3.average_data(df_real_columns)
    #    # HACK:2020.12.7:テストコード内でメインコードと同様の方法で算術平均を計算しているため、あまり好ましいテスト方法でない
    #    # 2. テストコード内で全区画算術平均を計算
    #    df_real_columns_average = (df_real_columns.describe()).loc['mean',:]
    #    print(df_real_columns_average, '\n')
    #    print(df_test)
    #    pd.testing.assert_series_equal(df_test, df_real_columns_average)

    def _test_average_data_in_all_section_and_return_dataframe(self):
        '''
        各columnsについて、全区間を算術平均し、計算結果をpd.DataFrame型オブジェクトで返したかテスト

        Return
        -----
        - 全区間を算術平均した場合、返り値は、pd.Seriesでなくpd.DataFrame。
        - 各列(特徴量)に、値が一つずつ含まれる状態。
        '''
        # 1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
        df_test = ap3.average_data(df_real_columns, len(df_real_columns), 'fixed_mean') # 全区間
        print(df_test, '\n')
        # 2. average_data関数の返り値の型がpd.DataFrameになっているかでアサーション
        self.assertIsInstance(df_test, pd.DataFrame)

    def _test_average_data_in_partly_section_and_return_dataframe(self):
        '''
        各columnsについて、部分的に区間を算術平均し、計算結果をpd.DataFrame型オブジェクトで返したかテスト
        '''
        mean_range = 5  # 平均値をとる要素数
        # 1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
        df_test = ap3.average_data(
                                input_acc_ang_df = df_real_columns, \
                                input_mean_range = mean_range, \
                                input_how = 'fixed_mean'
                                )
        print(df_test, '\n')
        # 2. [テスト1]: average_data関数の返り値の型がpd.DataFrameになっているかでアサーション
        self.assertIsInstance(df_test, pd.DataFrame)
        # 2. [テスト2]: average_data関数の返り値(↑pd.DataFrame型)の大きさが、(元のテストDataFrame型変数df_real_columnsの大きさ)/(mean_range)
        #    になっているかでアサーション
        self.assertEqual(len(df_test), int(len(df_real_columns)/mean_range))

    def _test_average_data_in_partly_section_and_return_dataframe_index_type_int(self):
        '''
        各columnsについて、部分的に区間を算術平均し、計算結果をpd.DataFrame型オブジェクトで返し、
        そのオブジェクトのインデックスオブジェクトの型がint型かどうかでテスト
        '''
        mean_range = 5  # 平均値をとる要素数
        # 1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
        df_test = ap3.average_data(
                                input_acc_ang_df = df_real_columns, \
                                input_mean_range = mean_range, \
                                input_how = 'fixed_mean'
                                )
        print('df_test')
        print('-----')
        print(df_test, '\n')
        print('df_test.index')
        print('-----')
        print(df_test.index, '\n')
        # 2. average_data関数の返り値のインデックスオブジェクトの型がintになっているかでアサーション
        #    インデックスオブジェクトの要素をランダムに抽出し、アサーション
        # HACK: 2021.12.8: もっと直接的な書き方があると思う。
        self.assertIsInstance(df_test.index[np.random.randint(len(df_test))], int)

    def _test_average_data_mean_range_1(self):
        '''
        main/ap3/average_data関数の引数について、input_mean_range=1を指定した場合、元のDataFrame型変数と値が変わらないかでテスト
        → ナイーブなやり方は、if input_mean_range=1: return input_df
        注1. 平均値を計算するにあたって, int型の要素はfloat型に変換される.
        注2. pd.DataFrame型変数のtime列は, str型のため, 平均を計算した際, 自動的に列ごと削除される.
        → 加速度/角速度の列のみを抽出して, 平均値の計算を行うのがベター.
        → 2020.12.16現在, csvから読み込んだpd.DataFrame型変数のインデックスは,
          Int64Index(~, dtype=\'int64\') (デフォルト) であるため好都合.(デフォルトでつけてくれるインデックスでOK)

        Notes
        -----
        - 関数average_dataの仕様について、
        　-- param: pd.Dataframeを'time'列ごと与える。固定平均については、'time'列の更新が含まれるため。
        　-- return: pd.Dataframeを返す。ただし、'time'列は列尾に追加。
        '''
        mean_range = 1                                              # テスト準備1: 平均値をとる要素数
        df_real = df_real_columns.loc[:, [                          # テスト準備2: 関数ap3.average_dataに与える引数
                                           'time',
                                           'Acceleration(X)[g]',
                                           'Acceleration(Y)[g]',
                                           'Acceleration(Z)[g]',
                                           'AngularRate(X)[dps]',
                                           'AngularRate(Y)[dps]',
                                           'AngularRate(Z)[dps]',
                                       ]]
        df_test = ap3.average_data(
                                input_acc_ang_df = df_real,     # 注. 'time'列ごと与えること
                                input_mean_range = mean_range,
                                input_how = 'slide_median',
                                )
        print('df_real\n{real}'.format(real=df_real))
        print('df_test\n{test}'.format(test=df_test))
        pd.testing.assert_frame_equal(df_real.drop('time', axis=1), df_test.drop('time', axis=1))   # 'time'列を除いてアサーション

    def _test_average_data_mean_range_1_df_read_is_generated_by_ap3_read_csv_(self):
        '''
        df_readを事前に用意するのでなく、ap3.read_csv_によりpd.DataFrame型変数を生成し、上テスト関数と同様のテストをする。

        Notes
        -----
        - 目的は、本番環境にできる限り近づけること。
        '''
        mean_range = 1                                              # テスト準備1: 平均値をとる要素数
        #df_real = ap3.read_csv_("./test_dataset/demo.csv")                                                              # テスト用データセット
        #df_real = ap3.read_csv_("../../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv")     # 本番用データセット
        df_real = ap3.read_csv_("../../dataset/32crossroad.csv").reset_index(drop='index')                              # 本番用データセット
        print(df_real)
        df_test = ap3.average_data(                         # 注. 'time'列ごと与えること
                                input_acc_ang_df = df_real.loc[:, 'time':'AngularRate(Z)[dps]'],
                                input_mean_range = mean_range,
                                input_how = 'slide_median',
                            )
        #print('df_real\n{real}'.format(real=df_real))
        #print('----------')
        #print('df_test\n{test}'.format(test=df_test))
        pd.testing.assert_frame_equal(df_real.loc[:, 'time':'AngularRate(Z)[dps]'].drop('time', axis=1),    \
                df_test.drop('time', axis=1))   # 'time'列を除いてアサーション

    def _test_average_data_index_type(self):
        '''
        ap3.average_data関数が返すpd.DataFrame型変数のインデックスオブジェクトの型がpd.Int64Indexかどうかでテスト
        '''
        mean_range = 3  # 平均値をとる要素数
        # 1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
        df_test = ap3.average_data(
                                input_acc_ang_df = df_real_columns.loc[:, 'time':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'fixed_mean',
                                )
        # 2. ap3.average_data関数が返すpd.DataFrame型変数のインデックスオブジェクトの型がpd.Int64Indexかどうかでアサーション
        #self.assertIsInstance(df_test.index, pd.Int64Index)    # 2021.12.8: HACK: いつの間にか、インデックスオブジェクトの型が変わっていた...
        self.assertIsInstance(df_test.index, pd.RangeIndex)

    def _test_average_data_input_how_raise_exception(self):
        '''
        input_howが不適切な値の場合、例外(Exception)を発生するかどうかでテスト
        '''
        mean_range = 3  # 平均値をとる要素数
        # 0. Exceptionオブジェクトが発生かどうかでアサーション
        with self.assertRaises(Exception):
        # 1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
            df_test = ap3.average_data(
                                    input_acc_ang_df = df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'], \
                                    input_mean_range = mean_range, \
                                    input_how = 'hoge-hoge',
                                    )

    def _test_average_data_fixed_mean_len(self):
        '''
        固定平均を算出した際、返り値のDataFrame型変数の大きさが適切かどうかテスト
        '''
        mean_range = 3  # 平均値をとる要素数
        # 1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
        df_test = ap3.average_data(
                                input_acc_ang_df = df_real_columns.loc[:, 'time':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'fixed_mean',   # 固定平均
                                )
        print(df_test, '\n')    # 「関数の出力値のDataFrame型変数」の値を出力
        # 2. 返り値のDataFrame型変数の大きさが、(引数のDataFrame型変数の大きさ)/(平均値幅)(→適切な固定平均の返り値の大きさ)
        #    になっているかでアサーション
        self.assertEqual(len(df_test), int(len(df_real_columns)/mean_range))

    def _test_average_data_slide_mean_len(self):
        '''
        移動平均を算出した際、返り値のDataFrame型変数の大きさが適切かどうかテスト
        '''
        mean_range = 10  # 平均値をとる要素数
        # 1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
        df_test = ap3.average_data(
                                input_acc_ang_df = df_real_columns, \
                                input_mean_range = mean_range, \
                                input_how = 'slide_mean',   # 移動平均
                                )
        print(df_test, '\n')    # 「関数の出力値のDataFrame型変数」の値を出力
        # 2. 返り値のDataFrame型変数の大きさが、(引数のDataFrame型変数の大きさ)-(平均値幅)+1(→適切な固定平均の返り値の大きさ)
        #    になっているかでアサーション'
        self.assertEqual(len(df_test), len(df_real_columns))

    def _test_average_data_fixed_slide_mean_val(self):
        '''
        固定/移動平均の算出結果について、値が正しいかテスト
        ＊完璧なテストではない
        '''
        mean_range = 3  # 平均値をとる要素数
        # 1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
        #    固定平均
        df_test_fixed = ap3.average_data(
                                input_acc_ang_df = df_real_columns.loc[:, 'time':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'fixed_mean',   # 固定平均
                                )
        print(df_test_fixed, '\n')
        # 2. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持
        #    移動平均'
        df_test_slide = ap3.average_data(
                                input_acc_ang_df = df_real_columns.loc[:, 'time':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'slide_mean',   # 移動平均
                                )
        print(df_test_slide, '\n')
        # 3. アサーション方法
        # 3-1. 固定/移動平均の返り値について、pd.DataFrame型変数の先頭行の値が一致しているかどうか
        # 3-2.    "   、pd.DataFrame型変数の先頭以降の行が一致していないかどうか
        # 3-3. 上記２つの条件を共に満たすかどうかでアサーション
        self.assertTrue(
                ((df_test_fixed.iloc[0]).equals(df_test_slide.iloc[0]))   # \
                & (not (df_test_fixed.iloc[1:]).equals(df_test_slide.iloc[1:]))  # \
                )


if __name__ == '__main__':
    unittest.main()
