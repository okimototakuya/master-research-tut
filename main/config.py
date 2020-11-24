'プロジェクトのグローバル変数を定義するクラス'
import sys
from pprint import pprint
from collections import OrderedDict
import pandas as pd


## HACK
## 方法1.python公式ドキュメント(https://docs.python.org/ja/3/faq/programming.html):グローバル変数モジュールのグローバル変数はカプセル化せず、剥き出し.
## 方法2.実践Python3:シングルトンデザインパターンでは、変数をプライベート化し、変数を取得するメソッドをパブリック化.
' 確率モデルによる予測値'
pred_by_prob_model = None

' 加速度データファイル(csv)のパス'
data_read_by_api = "../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16
#data_read_by_api = "../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19

' 加速度の方向名のリスト'
direct_acc = [
    'Acceleration_x',
    'Acceleration_y',
    'Acceleration_z',
    #'AngularRate_x',
    #'AngularRate_y',
    #'AngularRate_z',
    ]

' 時系列/加速度2次元プロット画像ファイルの保存先'
#save_graph_to_path = "/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100/"
save_graph_to_path = "/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hoge-hoge/"
#save_graph_to_path = "/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100/"
#save_graph_to_path = "/Users/okimototakuya/Desktop/tmp/"

' 1つのグラフにおけるプロット数'
#plot_amount_in_graph = 10000
plot_amount_in_graph = 131663

#' 加速度データファイルで、隠れマルコフモデルを適用させる範囲:始まり'
'加速度データファイル(csv)を抽出する範囲(first-last:リーダブルコードの包含関係をsedの挙動から確認済)'
data_sampled_first = 70000

' ":終わり'
data_sampled_last = 80000

' 加工した加速度データファイルを格納するDataFrame型変数'
data_sampled_by_func = None

'平均値をとる要素数'
mean_range = 1


##HACK:反例ありそう→データフレームの大きさと平均値幅のすり合わせ、例外処理
def aveData(input_dataframe):
    '加速度の平均値をとり、DataFrame型変数にして返す'
    # 加速度の平均値を格納するためのDataFrame型変数
    try:
        ave_dataframe = pd.DataFrame(index=[], columns=list(input_dataframe.columns))
        for i in range(int(len(input_dataframe)/mean_range)):
            #ave_dataframe = ave_dataframe.append(input_dataframe.iloc[i*mean_range:i*mean_range+mean_range, :].mean(), ignore_index=True)
            ave_dataframe = ave_dataframe.append(input_dataframe.iloc[i*mean_range:i*mean_range+mean_range, :].describe().loc['mean'], ignore_index=True)
    except ZeroDivisionError as err:
        print('平均値をとる要素数(mean_range)が０です.')
        sys.exit()
    except Exception as other:
        print('原因不明のエラーです.')
        sys.exit()
    return ave_dataframe


def main():
    print("pred_by_prob_model:", pred_by_prob_model)
    print("data_read_by_api:", data_read_by_api)
    print("direct_acc:", direct_acc)
    print("save_graph_to_path:", save_graph_to_path)
    print("plot_amount_in_graph:", plot_amount_in_graph)
    print("data_sampled_first:", data_sampled_first)
    print("data_sampled_last:", data_sampled_last)
    print("data_sampled_by_func:", data_sampled_by_func)

    #pprint(globals())
    #pprint(OrderedDict(globals()))

if __name__ == '__main__':
    main()
