import os
import sys
import pandas as pd
sys.path.append('../tests')
import test_acceleration_plot3 as tap3
import config


def read_csv_(input_path_to_csv):
    'csvファイル(加速度データ)を読み込み、pd.DataFrame型変数を返す関数  \
     引数1:csvファイルの相対パス'
    default_num_skip_row = 1    # 列名の行に欠損値 (None) が含まれるため、スキップし、列名をユーザが再定義(names)
    return pd.read_csv(
            input_path_to_csv,  # 入力のcsvファイルパス
            index_col = 0,  # 列0 (列名オブジェクトがNone) をインデックスに
            skiprows = tap3.TEST_DATA_SAMPLED_FIRST + default_num_skip_row,    \
                    # テスト:切り出し始め(line値TEST_DATA_SAMPLED_FIRSTはDataFrame型変数に含まれる)
            skipfooter = sum([1 for _ in open(input_path_to_csv)]) - (tap3.TEST_DATA_SAMPLED_LAST + default_num_skip_row),    \
                    # テスト:切り出し終わり(line値TEST_DATA_SAMPLED_LASTはDataFrame型変数に含まれない)
            #skiprows = config.data_sampled_first + default_num_skip_row,    \
            #        # 切り出し始め(line値config.data_sampled_firstはDataFrame型変数に含まれる)
            #skipfooter = sum([1 for _ in open(input_path_to_csv)]) - (config.data_sampled_last + default_num_skip_row),    \
            #        # 切り出し終わり(line値config.data_sampled_lastはDataFrame型変数に含まれない)
            header = None,
            names = ['Unnamed: 0', 'line', 'time',
                'Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]',
                'AngularRate(X)[dps]', 'AngularRate(Y)[dps]', 'AngularRate(Z)[dps]',
                'Temperature[degree]', 'Pressure[hPa]', 'MagnetCount', 'MagnetSwitch',
                'onCrossroad', 'crossroadID'],
            engine = 'python',
            )


def average_data(input_acc_ang_df, input_mean_range, input_how):
    'pd.DataFrame型変数を引数にし、特定区間における平均値を算出し、pd.DataFrame型変数を返す関数 \
     引数1:pd.DataFrame型変数の加速度/角速度の列(→pd.DataFrame型)    \
     引数2:平均値を計算する際の、要素数'
    if input_how == 'fixed_mean':  # 固定(?)平均
       len_after_division = int(len(input_acc_ang_df)/input_mean_range)    # 固定平均を算出した際、算出後のpd.DataFrame型変数の大きさ
       return pd.concat([(input_acc_ang_df.iloc[offset_i:offset_i+input_mean_range].describe()).loc['mean', :] \
               for offset_i in range(len_after_division)], axis=1).T.reset_index(drop='index') # インデックスオブジェクトの型はpd.Int64Index (pd.read_csvのデフォルト)
    elif input_how == 'slide_mean': # 移動平均
        pass
    elif input_how == 'slide_median':   # 移動中央値
        pass
    else:
        raise Exception('input_howに無効な値{wrong_input_how}が与えられています.'.format(wrong_input_how=input_how))


def main():
    df_read = read_csv_(config.data_read_by_api) # csvファイル(加速度データ)を読み込み、pd.DataFrame型変数を返す


if __name__ == '__main__':
    main()
