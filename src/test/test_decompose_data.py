import sys
import unittest
import numpy as np
import numpy.linalg as LA
import pandas as pd
sys.path.append('../main')
import acceleration_plot3 as ap3


class TestDecomposeData(unittest.TestCase):
    '''
    Factor Loadingなど、主成分分析の結果の妥当性をテスト
    '''
    def setUp(self):
        pass
        '''
        プロダクトコードmain関数内処理について、PCAを適用する前段階を再現
        '''
        #self.df_read = ap3.read_csv_('./test_dataset/demo.csv')         # 1. データのロード
        self.df_read = ap3.read_csv_(ap3.PATH_CSV_ACCELERATION_DATA)         # 1. データのロード
        self.df_averaged = ap3.average_data(                             # 2. 平均計算
                            input_acc_ang_df =                          # 引数1:pd.DataFrame型変数の加速度/角速度の列(→pd.DataFrame型)
                                    self.df_read.loc[:,[                # 行数(データ数)の指定
                                        'time',                         # 時刻
                                        'Acceleration(X)[g]',           # 列(特徴量)の指定
                                        'Acceleration(Y)[g]',
                                        'Acceleration(Z)[g]',
                                        'AngularRate(X)[dps]',
                                        'AngularRate(Y)[dps]',
                                        'AngularRate(Z)[dps]',
                                        ]],
                            input_mean_range = ap3.MEAN_RANGE, # 引数2:平均値を計算する際の、要素数
                            input_how = ap3.HOW_TO_CALCULATE_MEAN,   # 引数3:平均値の算出方法 fixed_mean:固定(?)平均, slide_mean:移動平均, slide_median:移動中央値
                                                                                                                    )

    def tearDown(self):
        pass

    def _test_decompose_data_match_singular_value_decomposition_by_scipy(self):
        '''
        scipyによる特異値分解と、ap3.decompose_data関数による主成分分析が一致するかテスト

        Notes
        -----
        - FIXME: 2021.11.9 21:30頃, Assertion Error: DataFrame are different
        '''
        df_pca = ap3.decompose_data(df_real_columns[['Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]']])
        df_test = df_real_columns
        pd.testing.assert_frame_equal(df_test, df_pca)

    def test_factor_loading_calculate_eig_value(self):
        '''
        Factor Loadingの行ベクトルの２乗和を計算し、固有値を算出
        '''
        df_pca, loading = ap3.decompose_data(self.df_averaged.drop('time', axis=1)) # Factor Loadingの算出
        print('df_pca.values\n', df_pca.values)
        for i in range(len(df_pca.values.T)):
            print('固有値{i}: {eig_val}'.format(i=i+1, eig_val=sum(np.square(loading[i]))))


if __name__ == '__main__':
    unittest.main()
