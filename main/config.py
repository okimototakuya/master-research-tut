'プロジェクトのグローバル変数を定義するクラス'
import sys
from pprint import pprint
from collections import OrderedDict
import pandas as pd


## HACK
## 方法1.python公式ドキュメント(https://docs.python.org/ja/3/faq/programming.html):グローバル変数モジュールのグローバル変数はカプセル化せず、剥き出し.
## 方法2.実践Python3:シングルトンデザインパターンでは、変数をプライベート化し、変数を取得するメソッドをパブリック化.
' 確率モデルによる予測値'
pred = None

' 加速度データファイル(csv)のパス'
filename = "../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16
#filename = "../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19

' 加速度の方向名のリスト'
acc = [
    'Acceleration_x',
    'Acceleration_y',
    'Acceleration_z',
    #'AngularRate_x',
    #'AngularRate_y',
    #'AngularRate_z',
    ]

' 時系列/加速度2次元プロット画像ファイルの保存先'
#path = "/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100"
path = "/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hoge-hoge"
#path = "/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100"
#path = "/Users/okimototakuya/Desktop/tmp"

' 1つのグラフにおけるプロット数'
plotseg = 10000
#plotseg = 131663

' 加速度データファイルで、隠れマルコフモデルを適用させる範囲:始まり'
hmmstart = 60000

' ":終わり'
hmmend = 69999

' 加工した加速度データファイルを格納するDataFrame型変数'
dataframe = None

'平均値をとる要素数'
AVERAGE = 10


##HACK:反例ありそう→データフレームの大きさと平均値幅のすり合わせ、例外処理
def aveData(input_dataframe):
    '加速度の平均値をとり、DataFrame型変数にして返す'
    # 加速度の平均値を格納するためのDataFrame型変数
    try:
        ave_dataframe = pd.DataFrame(index=[], columns=list(input_dataframe.columns))
        for i in range(int(len(input_dataframe)/AVERAGE)):
            ave_dataframe = ave_dataframe.append(input_dataframe.iloc[i*AVERAGE:i*AVERAGE+AVERAGE, :].mean(), ignore_index=True)
    except ZeroDivisionError as err:
        print('平均値をとる要素数(AVERAGE)が０です.')
        sys.exit()
    except Exception as other:
        print('原因不明のエラーです.')
        sys.exit()
    return ave_dataframe


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

if __name__ == '__main__':
    main()
