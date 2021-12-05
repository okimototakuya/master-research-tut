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
    '''
    Master_Research_Denso/main/acceleration_plot3.pyをテスト
    FIXME1:2020.11.27:test_save_dataframe_to_csv_とtest_read_csv_real_columnsを同時に走らせるとテストが通らない→一方のみだと通る
    FIXME1:2020.11.27:究極的にはtest_read_csv_real_columnsが通れば良いから、大した問題ではない
    '''

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _test_hmm_learn_data_is_return_type_dataframe(self):
        '''
        ap3.hmm_learn_data関数の返す値がpd.DataFrame型かどうかでテスト(回帰テスト)
        '''
        df_test = ap3.hmm_learn_data(df_real_columns[['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]']])
        self.assertIsInstance(df_test, pd.DataFrame)

    def _test_hmm_learn_data_is_return_type_ndarray(self):
        '''
        ap3.hmm_learn_data関数の返す値がnp.ndarray型かどうかでテスト(回帰テスト)
        '''
        df_test = ap3.hmm_learn_data(df_real_columns[['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]']])
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter(self):
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns, 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter2(self):
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns['Acceleration(X)[g]'], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter3(self):
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns.loc[:, ['Acceleration(X)[g]']], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter4(self):
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns.loc[:, ['Acceleration(X)[g]', 'Acceleration(Y)[g]']], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter5(self):
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns.loc[:, ['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]']], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter6(self):
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
        print(df_real_columns['Acceleration(X)[g]'])
        df_test = ap3.hmm_learn_data(ap3.average_data(df_real_columns.loc[:, ['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]',]], 1, 'fixed_mean'))
        self.assertIsInstance(df_test, np.ndarray)

    def _test_hmm_learn_data_in_ap3_average_data_parameter7(self):
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
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
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
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

    def _test_hmm_learn_data_in_ap3_average_data_parameter9(self):
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト

        Notes
        -----
        ver9でテストが通らなくなった。→ ['tuple' object has no attribute 'iloc']
        2021/6/24[解決]: df_inputの代入文の末尾に、余分に「,」が混入し、pd.DataFrame型でなく、tuple型で認識されてしまっていた。
        '''
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
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
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
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
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
        '''
        ap3.hmm_learn_data関数の引数について、ap3.average_data関数が返したpd.DataFrame型変数で動くかどうかテスト
        '''
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


if __name__ == '__main__':
    unittest.main()
