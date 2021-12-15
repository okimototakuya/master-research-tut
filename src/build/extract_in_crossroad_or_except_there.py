import sys
sys.path.append('../main')
import acceleration_plot3 as ap3
import pandas as pd


def main():
    '''
    '''
    #df_read = pd.read_csv(
    #        '../../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv'   # ID16児童
    #        )
    str_csv_path = '../../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv'  # ID16児童
    df_read = ap3.read_csv_(str_csv_path)
    #print(df_read)
    #print(df_read['onCrossroad']==0)
    #print(df_read[df_read['onCrossroad']==1])
    #df_except_crossroad = pd.DataFrame(columns=df_read.columns)
    #list_except_crossroad = []
    list_ = []
    for i in range(len(df_read.columns)):
        list_.append([])
    #print(df_except_crossroad)
    #print(list_except_crossroad)
    int_index_to_csv_start = 0      # 切り出し区間: 始め
    int_index_to_csv_end = 0        # 切り出し区間: 終わり
    num_csv = 0
    for i in range(len(df_read)):
        if df_read['onCrossroad'][i] == 0:                          # 交差点にいない。
            if i == 0 or df_read['onCrossroad'][i-1] == 1:          # - 一つ前の要素が交差点滞在(df_read['onCrossroad']==1)ならば、そこから。
                int_index_to_csv_start = i
            #df_except_crossroad.append(df_read.loc[i, :])          # HACK: 2021.12.15: 遅い (pd.DataFrame.append)
            #list_except_crossroad[i] = df_read.loc[i, :].tolist()
            for j in range(len(df_read.columns)):
                #list_[j] += df_read.iloc[i, j].tolist()
                #list_[j].append(df_read.iloc[i, j])        # appendによる要素の追加: やや速い
                #list_[j].extend([df_read.iloc[i, j]])       # extendによる要素の追加: とても速い
                list_[j].insert(0, [df_read.iloc[i, j]])       # insertによる要素の追加: とても速い
            #list_col.append(df_read.iloc[i, j]) for list_col in list_
        else:                                                       # 交差点にいる。
            if df_read['onCrossroad'][i-1] == 0:                    # - 一つ前の要素が交差点不在(df_read['onCrossroad']==0)ならば、そこまで。
                int_index_to_csv_end = i-1
                num_csv = num_csv + 1
                print('num_csv: ', num_csv)
            for j in range(len(df_read.columns)):
                #list_[j] += df_read.iloc[i, j].tolist()
                #list_[j].append(df_read.iloc[i, j])        # appendによる要素の追加: やや速い
                #list_[j].extend([df_read.iloc[i, j]])       # extendによる要素の追加: とても速い
                list_[j].insert(0, [df_read.iloc[i, j]])       # insertによる要素の追加: とても速い
                #df_except_crossroad.loc[int_index_to_csv_start:int_index_to_csv_end, :].to_csv('hoge-hoge{j}.csv'.format(j=j))
                pd.DataFrame({df_read.columns[k]: list_[k][int_index_to_csv_start:int_index_to_csv_end+1] for k in range(len(df_read.columns))}).to_csv('hoge-hoge{num_csv}.csv'.format(num_csv=num_csv))
            else:
                pass


if __name__ == '__main__':
    main()
