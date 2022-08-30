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


class TestReadCsv_(unittest.TestCase):
    '''
    関数read_csv_についてテスト

    Notes
    -----
    - HACK: 2021.12.2
    　-- csvファイルを読み込んで生成されたpd.DataFrame型変数(df_test)が、元のpd.DataFrame型変数(df_real_columns)と、
    　　 [たて × よこ]の大きさが変わらないことを確認した。
    　　--- 関数test_read_csv_read_right_length, 関数test_read_csv_read_right_columns_length
    　-- ただし、各列の型の不変性は確認していない。元々のプロダクトコードの処理からして本質でない。
    '''
    def setUp(self):
        '''
        テストcsvファイルを書込
        '''
        df_real_columns.to_csv('./test_dataset/demo.csv')
        #subprocess.call(['sed', '\'1', 's/,//\'', './test_dataset/demo.csv', '>', './test_dataset/demo.csv'])
        subprocess.getoutput('sed -i -e \'1 s/,//\' ./test_dataset/demo.csv')   # 書き出したテストcsvファイルの先頭行頭のカンマを削除

    def tearDown(self):
        '''
        次回のテストのためにテストcsvファイルを削除
        '''
        os.remove('./test_dataset/demo.csv')

    def _test_read_csv_read_right_length(self):
        '''
        テストcsvファイルをDataFrame型変数として正しい大きさで読み込めたかテスト

        Notes
        -----
        - 関数read_csv_の、skiprows, skipfooterの値のテストが目的
        　→ 2021.12.2: DATA_SAMPLED_FIRST, DATA_SAMPLE_LASTの定義を廃止した。
        　→ [理由1]: 交差点内の全区間をテスト対象とするという研究方針になったため、不必要となった。
        　→ [理由2]: プロダクトコードをやや煩雑にする要因になっていた。
        　　- グローバル変数AMOUNT_OF_PLOTの定義をする際、↑のグローバル変数に依存していた。(コードの密結合)
        　　- 全区間を解析するという方針になり、AMOUNT_OF_PLOTで置き換えられる箇所が大半であった。
        '''
        # テストcsvファイルを読込
        df_test = ap3.read_csv_('./test_dataset/demo.csv')
        #print('df_test')
        #print('-----')
        #print(df_test)
        #print('df_real_columns')
        #print('-----')
        #print(df_real_columns)
        #pd.testing.assert_frame_equal(df_test, df_real_columns)
        self.assertEqual(len(df_test), len(df_real_columns))

    def _test_read_csv_read_right_columns_length(self):
        '''
        テストcsvファイルをDataFrame型変数として正しい列数で読み込めたかテスト
        '''
        # テストcsvファイルを読込
        df_test = ap3.read_csv_('./test_dataset/demo.csv')
        #print('df_test')
        #print('-----')
        #print(df_test)
        #print('df_real_columns')
        #print('-----')
        #print(df_real_columns)
        #pd.testing.assert_frame_equal(df_test, df_real_columns)
        self.assertEqual(len(df_test.columns), len(df_real_columns.columns))

    def _test_read_csv_index_type(self):
        '''
        ap3.read_csv_関数が返すpd.DataFrame型変数のインデックスオブジェクトの型がpd.Int64Indexかどうかでテスト
        '''
        df_test = ap3.read_csv_('./test_dataset/demo.csv')
        # ap3.read_csv_関数が返すpd.DataFrame型変数のインデックスオブジェクトの型がpd.Int64Indexかどうかでアサーション
        self.assertIsInstance(df_test.index, pd.Int64Index)

    def _test_read_csv_data_type(self):
        '''
        ap3.read_csv_関数の返す値がpd.DataFrame型かどうかでテスト
        '''
        df_test = ap3.read_csv_('./test_dataset/demo.csv')
        # ap3.read_csv_関数の返す値がpd.DataFrame型かどうかでアサーション
        self.assertIsInstance(df_test, pd.DataFrame)

    def test_read_csv_match_size_read_dataframe_and_size_csv_file(self):
        '''
        読み込み対象のcsvファイルの大きさと、読み込み先のpd.DataFrame型の大きさが一致するかテスト
        '''
        file_path = './test_dataset/demo.csv'
        df_test = ap3.read_csv_(file_path)
        self.assertEqual(len(df_test), sum([1 for _ in open(file_path)]) - 1)   # 列名の行 (1行) 引いた。


if __name__ == '__main__':
    unittest.main()
