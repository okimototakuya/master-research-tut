'プロジェクトのグローバル変数を定義するクラス'
from pprint import pprint
from collections import OrderedDict
import pandas as pd


' 確率モデルによる予測値'
pred=None

' 加速度データファイル(csv)のパス'
filename="../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16
#filename="../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19

' 加速度の方向名のリスト'
acc=[
    'Acceleration_x',
    'Acceleration_y',
    'Acceleration_z',
    #'AngularRate_x',
    #'AngularRate_y',
    #'AngularRate_z',
    ]

' 時系列/加速度2次元プロット画像ファイルの保存先'
#path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100"
path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hoge-hoge"

#path="/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100"
#path="/Users/okimototakuya/Desktop/tmp"

' 1つのグラフにおけるプロット数'
plotseg=10000
#plotseg=131663

' 加速度データファイルで、隠れマルコフモデルを適用させる範囲:始まり'
hmmstart=60000

' ":終わり'
hmmend=69999

' 加工した加速度データファイルを格納するDataFrame型変数'
dataframe=None

'平均値をとる要素数'
AVERAGE = 10


def aveData(X):
    '加速度の平均値をとり、DataFrame型変数にして返す'
    # 加速度の平均値を格納するためのDataFrame型変数
    X_ave = pd.DataFrame(index=[], columns=acc)
    for i in range(int(len(X)/AVERAGE)):
        X_ave = X_ave.append(X.iloc[i*AVERAGE:i*AVERAGE+AVERAGE, :].mean(), ignore_index=True)
    return X_ave


def main():
    print("pred:", pred)
    print("filename:", filename)
    print("acc:", acc)
    print("path:", path)
    print("plotseg:", plotseg)
    print("hmmstart:", hmmstart)
    print("hmmend:", hmmend)
    print("dataframe:", dataframe)

    #pprint(globals())
    #pprint(OrderedDict(globals()))

if __name__ =='__main__':
    main()
