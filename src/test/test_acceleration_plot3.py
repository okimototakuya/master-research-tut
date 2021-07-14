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

    def setUp(self):
        pass

    def tearDown(self):
        pass

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
        print(df_real_columns[ap3.DATA_SAMPLED_FIRST:ap3.DATA_SAMPLED_LAST:1], '\n')
        print(df_test)
        pd.testing.assert_frame_equal(df_test, df_real_columns[ap3.DATA_SAMPLED_FIRST:ap3.DATA_SAMPLED_LAST:1])
        os.remove('./test_dataset/demo_sample.csv')   # 次回のテストのためにテストcsvファイルを削除

    def _test_read_csv_index_type(self):
        'ap3.read_csv_関数が返すpd.DataFrame型変数のインデックスオブジェクトの型がpd.Int64Indexかどうかでテスト'
        '1. テストcsvファイルを書込'
        df_real_columns.to_csv('./test_dataset/demo_sample.csv')
        '2. テストcsvファイルの一部を読込'
        df_test = ap3.read_csv_('./test_dataset/demo_sample.csv')
        '3. ap3.read_csv_関数が返すpd.DataFrame型変数のインデックスオブジェクトの型がpd.Int64Indexかどうかでアサーション'
        self.assertIsInstance(df_test.index, pd.Int64Index)
        os.remove('./test_dataset/demo_sample.csv')   # 次回のテストのためにテストcsvファイルを削除

    def _test_read_csv_data_type(self):
        'ap3.read_csv_関数の返す値がpd.DataFrame型かどうかでテスト'
        '1. テストcsvファイルを書込'
        df_real_columns.to_csv('./test_dataset/demo_sample.csv')
        '2. テストcsvファイルの一部を読込'
        df_test = ap3.read_csv_('./test_dataset/demo_sample.csv')
        '3. ap3.read_csv_関数の返す値がpd.DataFrame型かどうかでアサーション'
        self.assertIsInstance(df_test, pd.DataFrame)
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

    def _test_average_data_in_partly_section_and_return_dataframe(self):
        '各columnsについて、部分的に区間を算術平均し、計算結果をpd.DataFrame型オブジェクトで返したかテスト'
        mean_range = 5  # 平均値をとる要素数
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
        df_test = ap3.average_data(input_acc_ang_df = df_real_columns, \
                                input_mean_range = mean_range, \
                                )
        print(df_test, '\n')
        #'2. average_data関数の返り値の型がpd.DataFrameになっているかでアサーション'
        #self.assertIsInstance(df_test, pd.DataFrame)
        '2. average_data関数の返り値(↑pd.DataFrame型)の大きさが、(元のテストDataFrame型変数df_real_columnsの大きさ)/(mean_range)\
        になっているかでアサーション'
        self.assertEqual(len(df_test), int(len(df_real_columns)/mean_range))

    def _test_average_data_in_partly_section_and_return_dataframe_index_type_int(self):
        '各columnsについて、部分的に区間を算術平均し、計算結果をpd.DataFrame型オブジェクトで返し、\
        そのオブジェクトのインデックスオブジェクトの型がint型かどうかでテスト'
        mean_range = 5  # 平均値をとる要素数
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
        df_test = ap3.average_data(input_acc_ang_df = df_real_columns, \
                                input_mean_range = mean_range, \
                                )
        print(df_test, '\n')
        '2. average_data関数の返り値のインデックスオブジェクトの型がintになっているかでアサーション\
        → インデックスオブジェクトの要素をランダムに抽出し、アサーション'
        self.assertIsInstance(df_test.index[np.random.randint(len(df_test))], int)

    def _test_average_data_mean_range_1(self):
        'main/ap3/average_data関数の引数について、input_mean_range=1を指定した場合、元のDataFrame型変数と値が変わらないかでテスト\
        →ナイーブなやり方は、if input_mean_range=1: return input_df'
        '注1. 平均値を計算するにあたって, int型の要素はfloat型に変換される.'
        '注2. pd.DataFrame型変数のtime列は, str型のため, 平均を計算した際, 自動的に列ごと削除される.'
        '→ 加速度/角速度の列のみを抽出して, 平均値の計算を行うのがベター.'
        '→ 2020.12.16現在, csvから読み込んだpd.DataFrame型変数のインデックスは, \
           Int64Index(~, dtype=\'int64\') (デフォルト) であるため好都合.(デフォルトでつけてくれるインデックスでOK)'
        mean_range = 1  # 平均値をとる要素数
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
        df_test = ap3.average_data(
                                #input_acc_ang_df = df_real_columns, \
                                input_acc_ang_df = df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'fixed_mean',
                                )
        #print(df_real_columns, '\n')    # 「元のDataFrame型変数」の値を出力
        #print(df_real_columns.columns, '\n')    # 「元のDataFrame型変数」の列リストを出力
        #print(type(df_real_columns['time'][np.random.randint(AMOUNT_OF_ROW)]), '\n') # 「元のDataFrame型変数」のtime列の要素の型を出力
        print(df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'], '\n')    # 「元のDataFrame型変数」の加速度/角速度の列pd.DataFrameを出力
        print(df_test, '\n')                  # 「関数の出力値のDataFrame型変数」の値を出力
        #print(df_test.columns, '\n')  # 「関数の出力値のDataFrame型変数」の列リストを出力
        '2. 関数の出力値としてのDataFrame型変数と元のDataFrame型変数とで、値をアサーション'
        #pd.testing.assert_frame_equal(df_test, df_real_columns)
        pd.testing.assert_frame_equal(df_test, df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'])

    def _test_average_data_index_type(self):
        'ap3.average_data関数が返すpd.DataFrame型変数のインデックスオブジェクトの型がpd.Int64Indexかどうかでテスト'
        mean_range = 3  # 平均値をとる要素数
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
        df_test = ap3.average_data(
                                input_acc_ang_df = df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'fixed_mean',
                                )
        '2. ap3.average_data関数が返すpd.DataFrame型変数のインデックスオブジェクトの型がpd.Int64Indexかどうかでアサーション'
        self.assertIsInstance(df_test.index, pd.Int64Index)

    def _test_average_data_input_how_raise_exception(self):
        'input_howが不適切な値の場合、例外(Exception)を発生するかどうかでテスト'
        mean_range = 3  # 平均値をとる要素数
        '0. Exceptionオブジェクトが発生かどうかでアサーション'
        with self.assertRaises(Exception):
            '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
            df_test = ap3.average_data(
                                    input_acc_ang_df = df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'], \
                                    input_mean_range = mean_range, \
                                    input_how = 'hoge-hoge',
                                    )

    def _test_average_data_fixed_mean_len(self):
        '固定平均を算出した際、返り値のDataFrame型変数の大きさが適切かどうかテスト'
        mean_range = 3  # 平均値をとる要素数
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
        df_test = ap3.average_data(
                                input_acc_ang_df = df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'fixed_mean',   # 固定平均
                                )
        print(df_test, '\n')    # 「関数の出力値のDataFrame型変数」の値を出力
        '2. 返り値のDataFrame型変数の大きさが、(引数のDataFrame型変数の大きさ)/(平均値幅)(→適切な固定平均の返り値の大きさ)    \
            になっているかでアサーション'
        self.assertEqual(len(df_test), int(len(df_real_columns)/mean_range))

    def _test_average_data_slide_mean_len(self):
        '移動平均を算出した際、返り値のDataFrame型変数の大きさが適切かどうかテスト'
        mean_range = 3  # 平均値をとる要素数
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持'
        df_test = ap3.average_data(
                                input_acc_ang_df = df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'slide_mean',   # 移動平均
                                )
        print(df_test, '\n')    # 「関数の出力値のDataFrame型変数」の値を出力
        '2. 返り値のDataFrame型変数の大きさが、(引数のDataFrame型変数の大きさ)-(平均値幅)+1(→適切な固定平均の返り値の大きさ)    \
            になっているかでアサーション'
        self.assertEqual(len(df_test), len(df_real_columns)-mean_range+1)

    def _test_average_data_fixed_slide_mean_val(self):
        '固定/移動平均の算出結果について、値が正しいかテスト    \
         ＊完璧なテストではない'
        mean_range = 3  # 平均値をとる要素数
        '1. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持   \
            固定平均'
        df_test_fixed = ap3.average_data(
                                input_acc_ang_df = df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'fixed_mean',   # 固定平均
                                )
        print(df_test_fixed, '\n')
        '2. テストDataFrame型変数df_real_columnsを、ap3モジュール内average_data関数の引数にし、計算結果を保持   \
            移動平均'
        df_test_slide = ap3.average_data(
                                input_acc_ang_df = df_real_columns.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'], \
                                input_mean_range = mean_range, \
                                input_how = 'slide_mean',   # 移動平均
                                )
        print(df_test_slide, '\n')
        '3. アサーション方法    \
            3-1. 固定/移動平均の返り値について、pd.DataFrame型変数の先頭行の値が一致しているかどうか    \
            3-2.    "   、pd.DataFrame型変数の先頭以降の行が一致していないかどうか  \
            3-3. 上記２つの条件を共に満たすかどうかでアサーション'
        self.assertTrue(
                ((df_test_fixed.iloc[0]).equals(df_test_slide.iloc[0]))   # \
                & (not (df_test_fixed.iloc[1:]).equals(df_test_slide.iloc[1:]))  # \
                )

    def _test_hmm_learn_data_is_return_type_dataframe(self):
        'ap3.hmm_learn_data関数の返す値がpd.DataFrame型かどうかでテスト(回帰テスト)'
        df_test = ap3.hmm_learn_data(df_real_columns[['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]']])
        self.assertIsInstance(df_test, pd.DataFrame)

    def _test_hmm_learn_data_is_return_type_ndarray(self):
        'ap3.hmm_learn_data関数の返す値がnp.ndarray型かどうかでテスト(回帰テスト)'
        df_test = ap3.hmm_learn_data(df_real_columns[['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]']])
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns, 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter2(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns['Acceleration(X)[g]'], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter3(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns.loc[:, ['Acceleration(X)[g]']], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter4(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns.loc[:, ['Acceleration(X)[g]', 'Acceleration(Y)[g]']], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter5(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns.loc[:, ['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]']], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter6(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns.loc[:, ['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]',]], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter7(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(
                                            input_acc_ang_df =
                                                    df_real_columns.loc[:,[
                                                        'Acceleration(X)[g]',
                                                        'Acceleration(Y)[g]',
                                                        'Acceleration(Z)[g]',
                                                        ]],
                                            input_mean_range = 1,
                                            input_how = 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter8(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        print(df_real_columns['Acceleration(X)[g]'])
        df_averaged = ap3.average_data(
                            input_acc_ang_df =
                                    df_real_columns.loc[:,[
                                        'Acceleration(X)[g]',
                                        'Acceleration(Y)[g]',
                                        'Acceleration(Z)[g]',
                                        ]],
                            input_mean_range = 1,
                            input_how = 'fixed_mean')
        df_test = ap3.hmm_learn_data(df_averaged)
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter9(self):   # ver9でテストが通らなくなった。→ ['tuple' object has no attribute 'iloc']
        # 2021/6/24[解決]: df_inputの代入文の末尾に、余分に「,」が混入し、pd.DataFrame型でなく、tuple型で認識されてしまっていた。
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        df_input = df_real_columns.loc[:,[
                                        'Acceleration(X)[g]',
                                        'Acceleration(Y)[g]',
                                        'Acceleration(Z)[g]',
                                        ]]
        df_averaged = ap3.average_data(
                            input_acc_ang_df = df_input,
                            input_mean_range = 1,
                            input_how = 'fixed_mean')
        df_test = ap3.hmm_learn_data(df_averaged)
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter10(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        df_input = df_real_columns.loc[:,[
                                         'Acceleration(X)[g]',
                                         'Acceleration(Y)[g]',
                                         'Acceleration(Z)[g]',
                                         'AngularRate(X)[dps]',
                                         'AngularRate(Y)[dps]',
                                         'AngularRate(Z)[dps]'
                                        ]]
        df_averaged = ap3.average_data(
                            input_acc_ang_df =
                                    df_input.loc[:, [
                                        'Acceleration(X)[g]',
                                        'Acceleration(Y)[g]',
                                        'Acceleration(Z)[g]',
                                        #"AngularRate(X)[dps]",
                                        #"AngularRate(Y)[dps]",
                                        #"AngularRate(Z)[dps]",
                                        ]],
                            input_mean_range = 1,
                            input_how = 'fixed_mean'
                    )
        ndarray_test = ap3.hmm_learn_data(df_averaged)
        print(ndarray_test)
        self.assertIsInstance(ndarray_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter11(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        df_input_read_by_function_in_product_code = ap3.read_csv_(ap3.PATH_CSV_ACCELERATION_DATA)
        df_input_naive = df_real_columns.loc[:,[
                                         'Acceleration(X)[g]',
                                         'Acceleration(Y)[g]',
                                         'Acceleration(Z)[g]',
                                         'AngularRate(X)[dps]',
                                         'AngularRate(Y)[dps]',
                                         'AngularRate(Z)[dps]'
                                        ]]
        print(df_input_read_by_function_in_product_code)
        print(df_input_naive)
        df_averaged_input_read_by_function_in_product_code = ap3.average_data(
                            input_acc_ang_df =
                                    df_input_read_by_function_in_product_code.loc[:, [  # 通らない: pd.DataFrame型変数の作成の仕方を再現(→本番csvファイルを読込)
                                        'Acceleration(X)[g]',
                                        'Acceleration(Y)[g]',
                                        'Acceleration(Z)[g]',
                                        #"AngularRate(X)[dps]",
                                        #"AngularRate(Y)[dps]",
                                        #"AngularRate(Z)[dps]",
                                        ]],
                            input_mean_range = 1,
                            input_how = 'fixed_mean'
                    )
        df_averaged_input_naive = ap3.average_data(
                            input_acc_ang_df =
                                    df_input_naive.loc[:, [    # 通る: pd.DataFrame型変数をテスト用に自作
                                        'Acceleration(X)[g]',
                                        'Acceleration(Y)[g]',
                                        'Acceleration(Z)[g]',
                                        #"AngularRate(X)[dps]",
                                        #"AngularRate(Y)[dps]",
                                        #"AngularRate(Z)[dps]",
                                        ]],
                            input_mean_range = 1,
                            input_how = 'fixed_mean'
                    )
        print(df_averaged_input_read_by_function_in_product_code)
        print(df_averaged_input_naive)
        ndarray_test_input_read_by_function_in_product_code = ap3.hmm_learn_data(df_averaged_input_read_by_function_in_product_code)
        ndarray_test_input_naive = ap3.hmm_learn_data(df_averaged_input_naive)
        print(ndarray_test_input_read_by_function_in_product_code)
        print(ndarray_test_input_naive)
        self.assertIsInstance(ndarray_test_input_read_by_function_in_product_code, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter12(self):
        'ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト'
        df_input_read_by_function_in_product_code = ap3.read_csv_(ap3.PATH_CSV_ACCELERATION_DATA)
        print(df_input_read_by_function_in_product_code)
        df_averaged = ap3.average_data(
                            input_acc_ang_df =
                                    df_input_read_by_function_in_product_code.loc[:, [
                                        'Acceleration(X)[g]',
                                        'Acceleration(Y)[g]',
                                        'Acceleration(Z)[g]',
                                        #"AngularRate(X)[dps]",
                                        #"AngularRate(Y)[dps]",
                                        #"AngularRate(Z)[dps]",
                                        ]],
                            input_mean_range = 1,
                            input_how = 'fixed_mean'
                    )
        print(df_averaged)
        ndarray_test = ap3.hmm_learn_data(df_averaged)
        print(ndarray_test)
        self.assertIsInstance(ndarray_test, np.ndarray)

    def test_decompose_data(self):
        'scipyによる特異値分解と、ap3.decompose_data関数による主成分分析が一致するかテスト'
        df_pca = ap3.decompose_data(df_real_columns[['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]']])
        df_test = df_real_columns
        pd.testing.assert_frame_equal(df_test, df_pca)


if __name__ == '__main__':
    unittest.main()
