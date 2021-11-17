import sys
import unittest
sys.path.append('../main')
import acceleration_plot3 as ap3

class TestEstimateStateData(unittest.TestCase):
    '''
    HMMの予測結果の妥当性をテスト
    '''
    def setUp(self):
        '''
        プロダクトコードmain関数内処理について、HMMを適用する前段階を再現
        '''
        self.df_read = ap3.read_csv_(ap3.PATH_CSV_ACCELERATION_DATA) # 1. データのロード
        self.df_averaged = ap3.average_data(                             # 2. 平均計算
                            input_acc_ang_df =  # 引数1:pd.DataFrame型変数の加速度/角速度の列(→pd.DataFrame型)
                                    self.df_read.loc[:,[  # 行数(データ数)の指定
                                        'time',                 # 時刻
                                        'Acceleration(X)[g]',   # 列(特徴量)の指定
                                        'Acceleration(Y)[g]',
                                        'Acceleration(Z)[g]',
                                        'AngularRate(X)[dps]',
                                        'AngularRate(Y)[dps]',
                                        'AngularRate(Z)[dps]',
                                       ]],
                            input_mean_range = ap3.MEAN_RANGE, # 引数2:平均値を計算する際の、要素数
                            input_how = ap3.HOW_TO_CALCULATE_MEAN,   # 引数3:平均値の算出方法 fixed_mean:固定(?)平均, slide_mean:移動平均, slide_median:移動中央値
                    )
        self.df_pca = ap3.decompose_data(self.df_averaged.drop('time', axis=1)).join(self.df_averaged['time']) # 3. 主成分分析

    def tearDown(self):
        '''
        特になし
        '''
        pass

    def test_estimate_state_data_fitting_rate_predict_signal_between_original_and_pca(self):
        '''
        現データとPCA後データとの、状態ラベルの一致率を算出
        '''
        for i in range(10):
            dict_param_original = ap3.estimate_state_data(   # 主成分分析をせずに、隠れマルコフモデルを適用する場合
                    input_df_averaged = self.df_averaged.drop('time', axis=1),
                    input_how = ap3.ASSUMED_PROBABILISTIC_MODEL,
                    input_number_of_assumed_state = ap3.NUMBER_OF_ASSUMED_STATE,
                )
            dict_param_pca = ap3.estimate_state_data(   # 主成分分析をして、隠れマルコフモデルを適用する場合
                    input_df_averaged = self.df_pca.drop('time', axis=1),
                    input_how = ap3.ASSUMED_PROBABILISTIC_MODEL,
                    input_number_of_assumed_state = ap3.NUMBER_OF_ASSUMED_STATE,
                )
            num_fit = sum([1 if dict_param_original['状態系列の復号'][i] == dict_param_pca['状態系列の復号'][i] else 0 for i in range(len(dict_param_original['状態系列の復号']))])
            per_fit = num_fit / len(dict_param_original['状態系列の復号']) * 100
            print('試行{i}: 状態ラベルの一致率:{per_fit}'.format(i=i+1, per_fit=per_fit))


if __name__ == '__main__':
    unittest.main()
