import sys
import unittest
import numpy as np
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

    def tearDown(self):
        '''
        特になし
        '''
        pass

    def _test_estimate_state_data_return_state_sequence_which_size_is_input_df_averaged_length(self):
        '''
        関数estimate_state_dataに与えるdf_averagedと、HMM適用後のdict_param_original['状態系列の復号']とで、
        シーケンスの大きさが同じであることを確認
        '''
        dict_param_original = ap3.estimate_state_data(
                input_df_averaged = self.df_averaged.drop('time', axis=1),
                input_how = ap3.ASSUMED_PROBABILISTIC_MODEL,
                input_number_of_assumed_state = ap3.NUMBER_OF_ASSUMED_STATE,
            )
        self.assertEqual(len(self.df_averaged), len(dict_param_original['状態系列の復号']))

    def test_estimate_state_data_match_state_series_when_initial_state_and_random_seed_are_fixed(self):
        '''
        初期状態と乱数のシードを固定した場合(プロダクトコード内)、予測結果が変わらないことをテスト
        '''
        dict_param_original_1 = ap3.estimate_state_data(
                input_df_averaged = self.df_averaged.drop('time', axis=1),
                input_how = ap3.ASSUMED_PROBABILISTIC_MODEL,
                input_number_of_assumed_state = ap3.NUMBER_OF_ASSUMED_STATE,
            )
        print('dict_param_original_1の各キー列の型')
        print('-----')
        print([type(dict_param_original_1[key]) for key in dict_param_original_1.keys()])
        dict_param_original_2 = ap3.estimate_state_data(
                input_df_averaged = self.df_averaged.drop('time', axis=1),
                input_how = ap3.ASSUMED_PROBABILISTIC_MODEL,
                input_number_of_assumed_state = ap3.NUMBER_OF_ASSUMED_STATE,
            )
        #self.assertEqual(dict_param_original_1, dict_param_original_2)                         # 通らない: 辞書型変数の要素に、numpyサポートの型が含まれるため。
        #np.testing.assert_array_equal(dict_param_original_1, dict_param_original_2)            # 通らない: コンソールに出力された値そのものは同じに見えるが、AssertionErrorを返された。
        for key_1, key_2 in zip(dict_param_original_1.keys(), dict_param_original_2.keys()):
            np.testing.assert_array_equal(dict_param_original_1[key_1], dict_param_original_2[key_2])


if __name__ == '__main__':
    unittest.main()
