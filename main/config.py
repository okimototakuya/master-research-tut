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
#data_read_by_api = "../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16
#data_read_by_api = "../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19
data_read_by_api = "../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16(交差点ラベル付)
#data_read_by_api = "../dataset/labeledEditedLOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19(交差点ラベル付)

'グラフ描画に用いる特徴量(時系列/加速度2次元)'
features_selected_manually = [
    'time'
    'Acceleration(X)[g]',
    'Acceleration(Y)[g]',
    'Acceleration(Z)[g]',
    #'AngularRate(X)[dps]',
    #'AngularRate(Y)[dps]',
    #'AngularRate(Z)[dps]',
    ]

'PCA分析にかける特徴量'
features_analyzed_by_pca = [
    'Acceleration(X)[g]',
    'Acceleration(Y)[g]',
    'Acceleration(Z)[g]',
    #'AngularRate(X)[dps]',
    #'AngularRate(Y)[dps]',
    #'AngularRate(Z)[dps]',
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


def main():
    print("pred_by_prob_model:", pred_by_prob_model)
    print("data_read_by_api:", data_read_by_api)
    print("features_selected_manually:", features_selected_manually)
    print("save_graph_to_path:", save_graph_to_path)
    print("plot_amount_in_graph:", plot_amount_in_graph)
    print("data_sampled_first:", data_sampled_first)
    print("data_sampled_last:", data_sampled_last)
    print("data_sampled_by_func:", data_sampled_by_func)

    #pprint(globals())
    #pprint(OrderedDict(globals()))

if __name__ == '__main__':
    main()
