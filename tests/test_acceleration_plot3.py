import os
import sys
import glob
import subprocess
import datetime
import unittest
import numpy as np
import pandas as pd
sys.path.append('../main')
import acceleration_plot3 as ap3


AMOUNT_OF_ROW = 30  # テストcsvファイルの列数
'csvファイルを読み取る際の、切り出し区間'
TEST_DATA_SAMPLED_FIRST = 3    # 切り出し始め(line値TEST_DATA_SAMPLED_FIRSTはDataFrame型変数に含まれる)
TEST_DATA_SAMPLED_LAST = 7 # 切り出し終わり(line値TEST_DATA_SAMPLED_LASTはDataFrame型変数に含まれない)


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
        'Temperature[degree]':np.random.randn(AMOUNT_OF_ROW)+18,   # 正規分布に従って値を出力
        'Pressure[hPa]':np.random.randn(AMOUNT_OF_ROW)+1017,   # 正規分布に従って値を出力
        #'Temperature[degree]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        #'Pressure[hPa]':np.ones(AMOUNT_OF_ROW, dtype=int), # 常に値1を出力
        'MagnetCount':np.zeros(AMOUNT_OF_ROW, dtype=int), # 常に値0を出力
        'MagnetSwitch':np.zeros(AMOUNT_OF_ROW, dtype=int), # 常に値0を出力
        #'onCrossroad':0, # 常に値0を出力
        #'crossroadID':0, # 常に値0を出力
        'onCrossroad':np.random.binomial(1, 0.01, AMOUNT_OF_ROW), #ベルヌーイ分布に従ってブール値を出力
        'crossroadID':np.random.binomial(1, 0.01, AMOUNT_OF_ROW), #ベルヌーイ分布に従ってブール値を出力
    },
    )


class TestAccelerationPlot3(unittest.TestCase):
    'Master_Research_Denso/main/acceleration_plot3.pyをテスト'
    'FIXME1:2020.11.27:test_save_dataframe_to_csv_とtest_read_csv_real_columnsを同時に走らせるとテストが通らない→一方のみだと通る'
    'FIXME1:2020.11.27:究極的にはtest_read_csv_real_columnsが通れば良いから、大した問題ではない'

    #def _test_save_dataframe_to_csv_(self):
    #    'テストDataFrame型変数をテストcsvファイルに変換できたかテスト(テストコードのみの関数)'
    #    df_real_columns.to_csv('./test_dataset/demo.csv')
    #    #subprocess.call(['sed', '\'1', 's/,//\'', './test_dataset/demo.csv', '>', './test_dataset/demo.csv'])
    #    subprocess.getoutput('sed -i -e \'1 s/,//\' ./test_dataset/demo.csv')   # 書き出したテストcsvファイルの先頭行頭のカンマを削除
    #    self.assertTrue(glob.glob('./test_dataset/demo.csv'))

    #def _test_read_csv_one_column(self):
    #    'テストcsvファイルをDataFrame型変数として読み込めたかテスト(特徴量数1)'
    #    df_test = ap3.read_csv_('./test_dataset/demo.csv')
    #    df_one_column = pd.DataFrame({'a':[0]})
    #    pd.testing.assert_frame_equal(df_test, df_one_column)

    def _test_read_csv_real_columns(self):
        'テストcsvファイルをDataFrame型変数として読み込めたかテスト'
        '1. テストcsvファイルを書込'
        df_real_columns.to_csv('./test_dataset/demo.csv')
        '2. テストcsvファイルを読込'
        df_test = ap3.read_csv_('./test_dataset/demo.csv')
        #print(df_real_columns, '\n')
        #print(df_test)
        pd.testing.assert_frame_equal(df_test, df_real_columns)
        os.remove('./test_dataset/demo.csv')   # 次回のテストのためにテストcsvファイルを削除

    def _test_read_csv_real_columns_sample_partly(self):
        'テストcsvファイルの一部をDataFrame型変数として読み込めたかテスト'
        '1. テストcsvファイルを書込'
        df_real_columns.to_csv('./test_dataset/demo_sample.csv')
        '2. テストcsvファイルの一部を読込'
        df_test = ap3.read_csv_('./test_dataset/demo_sample.csv')
        print(df_real_columns[TEST_DATA_SAMPLED_FIRST:TEST_DATA_SAMPLED_LAST:1], '\n')
        print(df_test)
        pd.testing.assert_frame_equal(df_test, df_real_columns[TEST_DATA_SAMPLED_FIRST:TEST_DATA_SAMPLED_LAST:1])
        os.remove('./test_dataset/demo_sample.csv')   # 次回のテストのためにテストcsvファイルを削除

    #def _test_average_data_in_all_section_and_return_series_older(self):
    #    '各columnsについて、全区間を算術平均し、計算結果をpd.Series型オブジェクトで返したかテスト'
    #    '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
    #    df_test = ap3.average_data(df_real_columns)
    #    # HACK:2020.12.7:テストコード内でメインコードと同様の方法で算術平均を計算しているため、あまり好ましいテスト方法でない
    #    '2. テストコード内で全区画算術平均を計算'
    #    df_real_columns_average = (df_real_columns.describe()).loc['mean',:]
    #    print(df_real_columns_average, '\n')
    #    print(df_test)
    #    pd.testing.assert_series_equal(df_test, df_real_columns_average)

    def _test_average_data_in_all_section_and_return_series(self):
        '各columnsについて、全区間を算術平均し、計算結果をpd.Series型オブジェクトで返したかテスト'
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
        df_test = ap3.average_data(df_real_columns)
        print(df_test, '\n')
        '2. average_data関数の返り値の型がpd.Seriesになっているかでアサーション'
        self.assertIsInstance(df_test, pd.Series)

    def test_average_data_in_partly_section_and_return_dataframe(self):
        '各columnsについて、部分的に区間を算術平均し、計算結果をpd.DataFrame型オブジェクトで返したかテスト'
        mean_range = 5  # 平均値をとる要素数
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
        df_test = ap3.average_data(input_df = df_real_columns, \
                                input_mean_range = mean_range, \
                                )
        print(df_test, '\n')
        #'2. average_data関数の返り値の型がpd.DataFrameになっているかでアサーション'
        #self.assertIsInstance(df_test, pd.DataFrame)
        '2. average_data関数の返り値(↑pd.DataFrame型)の大きさが、(元のテストDataFrame型変数df_real_columnsの大きさ)/(mean_range)\
        になっているかでアサーション'
        self.assertEqual(len(df_test), int(len(df_real_columns)/mean_range))

    def test_average_data_in_partly_section_and_return_dataframe_index_type_int(self):
        '各columnsについて、部分的に区間を算術平均し、計算結果をpd.DataFrame型オブジェクトで返し、\
        そのオブジェクトのインデックスオブジェクトの型がint型かどうかでテスト'
        mean_range = 5  # 平均値をとる要素数
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
        df_test = ap3.average_data(input_df = df_real_columns, \
                                input_mean_range = mean_range, \
                                )
        print(df_test, '\n')
        '2. average_data関数の返り値のインデックスオブジェクトの型がintになっているかでアサーション\
        → インデックスオブジェクトの要素をランダムに抽出し、アサーション'
        self.assertIsInstance(df_test.index[np.random.randint(len(df_test))], int)


if __name__ == '__main__':
    unittest.main()
