import pandas as pd


def main():
    '''
    ID16, 19児童登下校ルートの、全交差点ラベルをピックアップするスクリプトファイル
    '''
    PATH = "../../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv"    # ID16
    df_read = pd.read_csv(
                    PATH,
                    index_col = 0,
                    header = None,
                    names = ['Unnamed: 0', 'line', 'time',
                        'Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]',
                        'AngularRate(X)[dps]', 'AngularRate(Y)[dps]', 'AngularRate(Z)[dps]',
                        'Temperature[degree]', 'Pressure[hPa]', 'MagnetCount', 'MagnetSwitch',
                        'onCrossroad', 'crossroadID'],
                    dtype = {'onCrossroad': int, 'crossroadID': int},   # 列'onCrossroad', 'crossroadID'をint型で読み込み
                    skiprows = 1,
                    engine = 'python',
                )
    cross_list = []                                                                             # 全交差点番号リスト
    for i in range(sum([1 for _ in open(PATH)])-1):                                             # 読み込んだファイルを全行探索
        if df_read['onCrossroad'][i] == 1 and df_read['crossroadID'][i] not in cross_list:
            cross_list.append(df_read['crossroadID'][i])                                        # 交差点番号を追加
    print('全交差点番号リスト:{list_}'.format(list_=cross_list))
    return cross_list


if __name__ == '__main__':
    main()
