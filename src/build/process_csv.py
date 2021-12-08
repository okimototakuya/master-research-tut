import glob
import pandas as pd

def main():
    '''
    csvファイルを加工し、岡田さんから頂いたcsvファイルのフォーマットに統一するためのスクリプト
    csvファイルの列頭に、列名を追加する。

    Notes
    -----
    - プロダクトコードの方で、関数read_csv_のオプションをいじるのは面倒。
    - ↑同時にテストコードの方も更新する必要があり、かなり面倒。
    '''
    #for name in glob.glob("../../dataset/hoge-hoge{num}.csv".format(num=0)):
    for name in glob.glob("./hoge-hoge?.csv"):
        df_read = pd.read_csv(
                    name,
                    names = ['Unnamed: 0', 'line', 'time',
                        'Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]',
                        'AngularRate(X)[dps]', 'AngularRate(Y)[dps]', 'AngularRate(Z)[dps]',
                        'Temperature[degree]', 'Pressure[hPa]', 'MagnetCount', 'MagnetSwitch',
                        'onCrossroad', 'crossroadID'],
                    )
        #print(df_read)
        df_read.to_csv(name)


if __name__ == '__main__':
    main()
