import os
import sys
import pandas as pd
sys.path.append('../tests')
import test_acceleration_plot3 as tap3


def read_csv_(input_path_to_csv):
    default_num_skip_row = 1    # 列名の行に欠損値 (None) が含まれるため、スキップし、列名をユーザが再定義(names)
    return pd.read_csv(
            input_path_to_csv,  # 入力のcsvファイルパス
            index_col = 0,  # 列0 (列名オブジェクトがNone) をインデックスに
            skiprows = tap3.TEST_DATA_SAMPLED_FIRST + default_num_skip_row,    \
                    # テスト:切り出し始め(line値TEST_DATA_SAMPLED_FIRSTはDataFrame型変数に含まれる)
            skipfooter = sum([1 for _ in open(input_path_to_csv)]) - (tap3.TEST_DATA_SAMPLED_LAST + default_num_skip_row),    \
                    # テスト:切り出し終わり(line値TEST_DATA_SAMPLED_LASTはDataFrame型変数に含まれない)
            header = None,
            names = ['Unnamed: 0', 'line', 'time',
                'Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]',
                'AngularRate(X)[dps]', 'AngularRate(Y)[dps]', 'AngularRate(Z)[dps]',
                'Temperature[degree]', 'Pressure[hPa]', 'MagnetCount', 'MagnetSwitch',
                'onCrossroad', 'crossroadID'],
            )


def main():
    pass


if __name__ == '__main__':
    main()
