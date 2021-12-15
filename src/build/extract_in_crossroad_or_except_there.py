import sys
sys.path.append('../main')
import acceleration_plot3 as ap3
import pandas as pd


def main():
    '''
    交差点不在/滞在時のデータ点を抽出し、csvファイルとしてエクスポートするための
    Pythonスクリプト
    '''
    str_csv_path = '../../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv'  # ID16児童
    df_read = ap3.read_csv_(str_csv_path)
    int_index_to_csv_start = 0      # 切り出し区間: 始め
    int_index_to_csv_end = 0        # 切り出し区間: 終わり
    num_csv = 0
    for i in range(len(df_read)):
        if df_read['onCrossroad'][i] == 0:                          # 交差点にいない。
            if i == 0 or df_read['onCrossroad'][i-1] == 1:          # - 一つ前の要素が交差点滞在(df_read['onCrossroad']==1)ならば、そこから。
                int_index_to_csv_start = i
            elif i == len(df_read) - 1:
                num_csv = num_csv + 1
                print('num_csv: ', num_csv)
                dict_ = {df_read.columns[j]: df_read.loc[int_index_to_csv_start:int_index_to_csv_end, df_read.columns[j]].tolist() for j in range(len(df_read.columns))}
                pd.DataFrame(data=dict_, columns=df_read.columns).to_csv('../test/test-build/hoge_hoge{num_csv}.csv'.format(num_csv=num_csv))
            else:
                pass
        else:                                                       # 交差点にいる。
            if df_read['onCrossroad'][i-1] == 0:                    # - 一つ前の要素が交差点不在(df_read['onCrossroad']==0)ならば、そこまで。
                num_csv = num_csv + 1
                print('num_csv: ', num_csv)
                dict_ = {df_read.columns[j]: df_read.loc[int_index_to_csv_start:int_index_to_csv_end-1, df_read.columns[j]].tolist() for j in range(len(df_read.columns))}
                pd.DataFrame(data=dict_, columns=df_read.columns).to_csv('../test/test-build/hoge_hoge{num_csv}.csv'.format(num_csv=num_csv))
            else:
                pass
        int_index_to_csv_end += 1


if __name__ == '__main__':
    main()
