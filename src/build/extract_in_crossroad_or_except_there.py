import sys
sys.path.append('../main')
import acceleration_plot3 as ap3
import pandas as pd


num_csv = 0                     # エクスポートしたcsvファイルの個数
bool_is_oncrossroad = True     # True: 滞在時, False: 不在時
path_to_export = '../test/test-build/hoge_hoge{num_csv}.csv'

def export_to_csv(df_read, int_index_to_csv_start, int_index_to_csv_end, bool_is_eof):
    '''
    pd.DataFrame型変数を、csvファイルにエクスポートするための関数

    Notes
    -----
    - エクスポート先のパス: path_to_export
    '''
    global num_csv
    global path_to_export
    num_csv = num_csv + 1
    print('num_csv: ', num_csv)
    end_index = 0 if bool_is_eof == True else 1     # csvファイル終端を調整: 最後にエクスポートするcsvファイルのみ別処理(off-by-oneエラーの防止)
    dict_to_csv = {df_read.columns[j]: df_read.loc[int_index_to_csv_start:int_index_to_csv_end - end_index, df_read.columns[j]].tolist() for j in range(len(df_read.columns))}
    #pd.DataFrame(data=dict_to_csv, index=list(dict_to_csv.values())[0]).to_csv(path_to_export.format(num_csv=num_csv))
    pd.DataFrame(data=dict_to_csv).to_csv(path_to_export.format(num_csv=num_csv))

def main():
    '''
    交差点不在/滞在時のデータ点を抽出し、csvファイルとしてエクスポートするための
    Pythonスクリプト

    Notes
    -----
    - 交通時運動データについて、最初/最後の点は交差点不在と仮定する。
    '''
    global bool_is_oncrossroad
    str_csv_path = '../../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv'  # ID16児童
    df_read = ap3.read_csv_(str_csv_path)
    int_index_to_csv_start = 0      # 切り出し区間: 始め
    int_index_to_csv_end = 0        # 切り出し区間: 終わり
    for i in range(len(df_read)):
        if df_read['onCrossroad'][i] == int(bool_is_oncrossroad):                       # [条件1]
            if i == 0 or df_read['onCrossroad'][i-1] == int(not bool_is_oncrossroad):   # - 一つ前の要素が交差点滞在(df_read['onCrossroad']==1)ならば、そこから。
                int_index_to_csv_start = i                                              # - また、最初の点は交差点不在と仮定。
            elif i == len(df_read) - 1:                                                 # -  "    最後  "                   。
                export_to_csv(df_read, int_index_to_csv_start, int_index_to_csv_end, True)
            else:
                pass
        else:                                                                           # [条件2]
            if i == 0:
                pass
            elif df_read['onCrossroad'][i-1] == int(bool_is_oncrossroad):                 # - 一つ前の要素が交差点不在(df_read['onCrossroad']==0)ならば、そこまで。
                export_to_csv(df_read, int_index_to_csv_start, int_index_to_csv_end, False)
            else:
                pass
        int_index_to_csv_end += 1                                   # 毎回のループごとに、抽出するcsvファイルの終端インデックスを1加算


if __name__ == '__main__':
    main()
