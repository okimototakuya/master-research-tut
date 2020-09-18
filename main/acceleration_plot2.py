import os
import sys
import subprocess as sp
import pandas as pd
import dask.dataframe as dd
#matplotlib.use('Agg')  # pyplotで生成した画像を保存するためのインポート
import matplotlib.pyplot as plt
import numpy as np
import hmm_learn
import cluster_learn


class Global():
    'acceleration_plot2モジュールのグローバル変数を属性に持つクラス'
    ' 確率モデルによる予測値'
    pred=None,
    ' 加速度データファイル(csv)のパス'
    filename="../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv",  # ID16
    #filename="../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv",  # ID19
    ' 加速度の方向名のリスト'
    acc=[
        'Acceleration_x',
        'Acceleration_y',
        'Acceleration_z',
        #'AngularRate_x',
        #'AngularRate_y',
        #'AngularRate_z',
        ],
    ' 時系列/加速度2次元プロット画像ファイルの保存先'
    #path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100",
    path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hoge-hoge",
    #path="/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100",
    #path="/Users/okimototakuya/Desktop/tmp",
    ' 1つのグラフにおけるプロット数'
    plotseg=10000,
    #plotseg=131663,
    ' 加速度データファイルで、隠れマルコフモデルを適用させる範囲:始まり'
    hmmstart=60000,
    ' ":終わり'
    hmmend=69999,
    ' 加工した加速度データファイルを格納するDataFrame型変数'
    dataframe=None,

    'Globalオブジェクト作成時に使用'
    #def __init__(self, **kwargs):
    #    self.hidden_pred = kwargs["input_pred"]
    #    self.hidden_filename = kwargs["input_filename"]
    #    self.hidden_acc = kwargs["input_acc"]
    #    self.hidden_path = kwargs["input_path"]
    #    self.hidden_plotseg = kwargs["input_plotseg"]
    #    self.hidden_hmmstart = kwargs["input_hmmstart"]
    #    self.hidden_hmmend = kwargs["input_hmmend"]
    #    self.hidden_dataframe = kwargs["input_dataframe"]

    #'ゲッターメソッド'
    #def get_pred(self):
    #    return self.hidden_pred
    #def get_filename(self):
    #    return self.hidden_filename
    #def get_acc(self):
    #    return self.hidden_acc
    #def get_path(self):
    #    return self.hidden_path
    #def get_plotseg(self):
    #    return self.hidden_plotseg
    #def get_hmmstart(self):
    #    return self.hidden_hmmstart
    #def get_hmmend(self):
    #    return self.hidden_hmmend
    #def get_dataframe(self):
    #    return self.hidden_dataframe

    #'セッターメソッド'
    ## 確率モデルによる予測値
    #def set_pred(self, input_pred):
    #    self.hidden_pred = input_pred
    ## 加速度データファイル(csv)のパス
    #def set_filename(self, input_filename):
    #    self.hidden_filename = input_filename
    ## 加速度の方向名のリスト
    #def set_acc(self, input_acc):
    #    self.hidden_acc = input_acc
    ## 時系列/加速度2次元プロット画像ファイルの保存先
    #def set_path(self, input_path):
    #    self.hidden_path = input_path
    ## 1つのグラフにおけるプロット数
    #def set_plotseg(self, input_plotseg):
    #    self.hidden_plotseg = input_plotseg
    ## 加速度データファイルで、隠れマルコフモデルを適用させる範囲:始まり
    #def set_hmmstart(self, input_hmmstart):
    #    self.hidden_hmmstart = input_hmmstart
    ## ":終わり
    #def set_hmmend(self, input_hmmend):
    #    self.hidden_hmmend = input_hmmend
    ## 加工した加速度データファイルを格納するDataFrame型変数
    #def set_dataframe(self, input_dataframe):
    #    self.hidden_dataframe = input_dataframe

    #'プロパティ'
    #pred = property(get_pred, set_pred)
    #filename = property(get_filename, set_filename)
    #acc = property(get_acc, set_acc)
    #path = property(get_path, set_path)
    #plotseg = property(get_plotseg, set_plotseg)
    #hmmstart = property(get_hmmstart, set_hmmstart)
    #hmmend = property(get_hmmend, set_hmmend)
    #dataframe = property(get_dataframe, set_dataframe)


class DataframeMaker():
    'excelファイルを読み込み、DataFrame型変数を生成する'
    def __init__(self, filename, acc, hmmstart, hmmend):
        # 列名を明示的に指定することにより, 欠損値をNaNで補完.
        col_names = [
            'line', 'time',
            'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
            'AngularRate_x', 'AngularRate_y', 'AngularRate_z',
            'Temperture', 'Pressure', 'MagnetCount', 'MagnetSwitch',
            ]
        self.df = dd.read_csv(
            filename,
            names=col_names,
            parse_dates=['time'],
            #index_col='time',
            #skiprows=3,
            #skiprows=[3],
            #skiprows=lambda x: x not in [i for i in range(hmmstart+3, hmmend+3)],
            converters={
                'line':int, 'time':str,
                'Acceleration_x':float, 'Acceleration_y':float, 'Acceleration_z':float,
                'AngularRate_x':float, 'AngularRate_y':float, 'AngularRate_z':float,
                'Temperture':float, 'Pressure':float, 'MagnetCount':int, 'MagnetSwitch':int,
                },
            #usecols=lambda x: x in acc+[index_col],
            usecols=lambda x: x in acc+['time'],
            ).compute()

    @staticmethod
    def cut_csv(buf, filename, hmmstart, hmmend):
        cmd1 = "sed -e 1,3d {filename}".format(filename=filename)
        cmd2 = "sed -n {start},{end}p".format(start=hmmstart, end=hmmend)
        res1 = sp.Popen(cmd1.split(" "), stdout=sp.PIPE)
        with open(buf, 'w') as f:
            sp.Popen(cmd2.split(" "), stdin=res1.stdout, stdout=f)
        #sp.run(['awk', '-F', '","', '{print}'])


class DataframePlotter():

    'DataFrameMakerクラスから生成したDataFrame型変数をプロットする'
    def __init__(self):
        pass

    @staticmethod
    def time_pred_plot(df, delta, args):
        predict = pd.DataFrame(pred, columns=['pred'])
        df = pd.concat([df[list(args)], predict], axis=1)
        ## 加速度・角速度の時系列変化をプロット
        for i in range(int(len(df)/delta)):
            copy_df = df.loc[delta*i:delta*(i+1), :]
            copy_df.dropna(how='all')
            ax1 = copy_df[list(args)].plot()
            ax = copy_df[['pred']].plot(ax=ax1)
            ax.set_title(filename)
            ax.set_ylim([-5.0, 2.5])
            plt.show()
            #plt.savefig(os.path.join(PATH, "demo"+str(i)+".png"))

    @staticmethod
    def acc1_acc2_plot(df):
        '加速度の2次元データをプロットする'
        ax = df.plot.scatter(x=acc[0], y=acc[1])   # 散布図
        ax.set_title(filename)
        ax.set_xlim([-5.5, 1.0])
        ax.set_ylim([-2.5, 2.0])
        plt.show()

    @staticmethod
    def plot(df, delta, args):  # delta:グラフの定義域,*args:グラフを描く列のタプル(＊タプルで受け取る)
        'DataFrame型変数をプロットする'
        global pred
        df = df.iloc[HMM_RANGE_START:HMM_RANGE_END, :].reset_index()
        df = hmm_learn.aveData(df)  # 加速度データを平均化
        delta = int(delta/hmm_learn.AVERAGE)    # 平均値をとる要素数で区間を割る
        if sys.argv[1] != '2':  # 隠れマルコフモデルorクラスタリングの時系列データを表示
            DataframePlotter.time_pred_plot(df, delta, args)
        else:   # 加速度データを2次元プロット
            DataframePlotter.acc1_acc2_plot(df)


def main():
    '確率モデルを適用し、学習結果を時系列表示する'
    'もしくは、加速度データを2次元プロットする'

    'グローバル変数のセット'
    #global_parameter = Global(
    #    input_pred=None,
    #    input_filename="../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv",  # ID16
    #    #input_filename="../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv",  # ID19
    #    input_acc=[
    #        'Acceleration_x',
    #        'Acceleration_y',
    #        'Acceleration_z',
    #        #'AngularRate_x',
    #        #'AngularRate_y',
    #        #'AngularRate_z',
    #        ],
    #    #input_path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100",
    #    input_path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hoge-hoge",
    #    #input_path="/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100",
    #    #input_path="/Users/okimototakuya/Desktop/tmp",
    #    input_plotseg=10000,
    #    #input_plotseg=131663,
    #    input_hmmstart=60000,
    #    input_hmmend=69999,
    #    input_dataframe=None,
    #    )

    '加速度データのDataFrame型変数を属性とする、DataframeMaker型オブジェクトを作成'
    buf = "../dataset/buf.csv"  # 加工したcsvファイルの保存先
    DataframeMaker.cut_csv(buf, Global().filename[0], Global().hmmstart[0], Global().hmmend[0])
    dataframe = DataframeMaker(buf, Global().acc[0], Global().hmmstart[0], Global().hmmend[0])

    'メインプログラム実行時の引数によって、描画するグラフを決定'
    if sys.argv[1] == '0':    # 隠れマルコフモデル
        #np.set_printoptions(threshold=np.inf)    # 配列の要素を全て表示(状態系列)
        hmm_learn.hmmLearn(dataframe.df)
        #pred = hmm_learn.pred
    elif sys.argv[1] == '1':    # クラスタリング
        #np.set_printoptions(threshold=np.inf)    # 配列の要素を全て表示(状態系列)
        cluster_learn.clusterLearn(dataframe.df)
        #pred = cluster_learn.pred
    elif sys.argv[1] == '2':    # 加速度を２次元プロット
        pass
    else:
        print("Error is here.")

    'グラフを描画'
    #DataframePlotter.plot(dataframe.df, PLOT_SEG, tuple(acc))
    DataframePlotter.plot(dataframe.df, Global().plotseg[0], tuple(Global().acc[0]))

if __name__ == '__main__':
    main()
