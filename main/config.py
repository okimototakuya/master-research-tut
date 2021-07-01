'プロジェクトのグローバル変数を定義するクラス'
import sys
from pprint import pprint
from collections import OrderedDict
import pandas as pd


## HACK
## 方法1.python公式ドキュメント(https://docs.python.org/ja/3/faq/programming.html):グローバル変数モジュールのグローバル変数はカプセル化せず、剥き出し.
## 方法2.実践Python3:シングルトンデザインパターンでは、変数をプライベート化し、変数を取得するメソッドをパブリック化.
' 1つのグラフにおけるプロット数'
#plot_amount_in_graph = 10000
plot_amount_in_graph = 131663


def main():
    print("plot_amount_in_graph:", plot_amount_in_graph)
    #pprint(globals())
    #pprint(OrderedDict(globals()))


if __name__ == '__main__':
    main()
