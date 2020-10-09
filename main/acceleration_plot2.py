import os
import sys
import subprocess as sp
import pandas as pd
import dask.dataframe as dd
import matplotlib
#matplotlib.use('Agg')  # pyplotで生成した画像を保存するためのインポート
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import config
import hmm_learn
import cluster_learn


##################################################################
##########グローバル変数クラス####################################
##################################################################
#class Global():
#    'acceleration_plot2モジュールのグローバル変数を属性に持つクラス'
#    #' 確率モデルによる予測値'
#    #pred_by_prob_model=None,
#    ' 加速度データファイル(csv)のパス'
#    data_read_by_api="../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv",  # ID16
#    #data_read_by_api="../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv",  # ID19
#    ' 加速度の方向名のリスト'
#    direct_acc=[
#        'Acceleration_x',
#        'Acceleration_y',
#        'Acceleration_z',
#        #'AngularRate_x',
#        #'AngularRate_y',
#        #'AngularRate_z',
#        ],
#    ' 時系列/加速度2次元プロット画像ファイルの保存先'
#    #save_graph_to_path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100",
#    save_graph_to_path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hoge-hoge",
#    #save_graph_to_path="/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100",
#    #save_graph_to_path="/Users/okimototakuya/Desktop/tmp",
#    ' 1つのグラフにおけるプロット数'
#    plot_amount_in_graph=10000,
#    #plot_amount_in_graph=131663,
#    ' 加速度データファイルで、隠れマルコフモデルを適用させる範囲:始まり'
#    data_sampled_first=60000,
#    ' ":終わり'
#    data_sampled_last=69999,
#    ' 加工した加速度データファイルを格納するDataFrame型変数'
#    #data_sampled_by_func=None,
#
#    'Globalオブジェクト作成時に使用'
#    #def __init__(self, **kwargs):
#    #    self.hidden_pred_by_prob_model = kwargs["input_pred_by_prob_model"]
#    #    self.hidden_data_read_by_api = kwargs["input_data_read_by_api"]
#    #    self.hidden_direct_acc = kwargs["input_direct_acc"]
#    #    self.hidden_save_graph_to_path = kwargs["input_save_graph_to_path"]
#    #    self.hidden_plot_amount_in_graph = kwargs["input_plot_amount_in_graph"]
#    #    self.hidden_data_sampled_first = kwargs["input_data_sampled_first"]
#    #    self.hidden_data_sampled_last = kwargs["input_data_sampled_last"]
#    #    self.hidden_data_sampled_by_func = kwargs["input_data_sampled_by_func"]
#
#    #'ゲッターメソッド'
#    #def get_pred_by_prob_model(self):
#    #    return self.hidden_pred_by_prob_model
#    #def get_data_read_by_api(self):
#    #    return self.hidden_data_read_by_api
#    #def get_direct_acc(self):
#    #    return self.hidden_direct_acc
#    #def get_save_graph_to_path(self):
#    #    return self.hidden_save_graph_to_path
#    #def get_plot_amount_in_graph(self):
#    #    return self.hidden_plot_amount_in_graph
#    #def get_data_sampled_first(self):
#    #    return self.hidden_data_sampled_first
#    #def get_data_sampled_last(self):
#    #    return self.hidden_data_sampled_last
#    #def get_data_sampled_by_func(self):
#    #    return self.hidden_data_sampled_by_func
#
#    #'セッターメソッド'
#    ## 確率モデルによる予測値
#    #def set_pred_by_prob_model(self, input_pred_by_prob_model):
#    #    self.hidden_pred_by_prob_model = input_pred_by_prob_model
#    ## 加速度データファイル(csv)のパス
#    #def set_data_read_by_api(self, input_data_read_by_api):
#    #    self.hidden_data_read_by_api = input_data_read_by_api
#    ## 加速度の方向名のリスト
#    #def set_direct_acc(self, input_direct_acc):
#    #    self.hidden_direct_acc = input_direct_acc
#    ## 時系列/加速度2次元プロット画像ファイルの保存先
#    #def set_save_graph_to_path(self, input_save_graph_to_path):
#    #    self.hidden_save_graph_to_path = input_save_graph_to_path
#    ## 1つのグラフにおけるプロット数
#    #def set_plot_amount_in_graph(self, input_plot_amount_in_graph):
#    #    self.hidden_plot_amount_in_graph = input_plot_amount_in_graph
#    ## 加速度データファイルで、隠れマルコフモデルを適用させる範囲:始まり
#    #def set_data_sampled_first(self, input_data_sampled_first):
#    #    self.hidden_data_sampled_first = input_data_sampled_first
#    ## ":終わり
#    #def set_data_sampled_last(self, input_data_sampled_last):
#    #    self.hidden_data_sampled_last = input_data_sampled_last
#    ## 加工した加速度データファイルを格納するDataFrame型変数
#    #def set_data_sampled_by_func(self, input_data_sampled_by_func):
#    #    self.hidden_data_sampled_by_func = input_data_sampled_by_func
#
#    #'プロパティ'
#    #pred_by_prob_model = property(get_pred_by_prob_model, set_pred_by_prob_model)
#    #data_read_by_api = property(get_data_read_by_api, set_data_read_by_api)
#    #direct_acc = property(get_direct_acc, set_direct_acc)
#    #save_graph_to_path = property(get_save_graph_to_path, set_save_graph_to_path)
#    #plot_amount_in_graph = property(get_plot_amount_in_graph, set_plot_amount_in_graph)
#    #data_sampled_first = property(get_data_sampled_first, set_data_sampled_first)
#    #data_sampled_last = property(get_data_sampled_last, set_data_sampled_last)
#    #data_sampled_by_func = property(get_data_sampled_by_func, set_data_sampled_by_func)


##################################################################
##########例外クラス##############################################
##################################################################
class WrongArgumentException(Exception):
    'pythonスクリプトに与えられた引数が適切でない時に返す例外'
    pass


##################################################################
##########データフレーム型変数を作るクラス########################
##################################################################
class DataframeMaker():
    '加速度データファイルを読み込み、DataFrame型変数を生成する'
    def __init__(self, data_read_by_api, direct_acc, data_sampled_first, data_sampled_last):
        # 列名を明示的に指定することにより, 欠損値をNaNで補完.
        col_names = [
            'line', 'time',
            'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
            'AngularRate_x', 'AngularRate_y', 'AngularRate_z',
            'Temperture', 'Pressure', 'MagnetCount', 'MagnetSwitch',
            ]
        self.df = pd.read_csv(
        #self.df = dd.read_csv(
            data_read_by_api,
            names=col_names,
            parse_dates=['time'],
            #index_col='time',
            #skiprows=3,
            #skiprows=[3],
            #skiprows=lambda x: x not in [i for i in range(data_sampled_first+3, data_sampled_last+3)],
            converters={
                'line':int, 'time':str,
                'Acceleration_x':float, 'Acceleration_y':float, 'Acceleration_z':float,
                'AngularRate_x':float, 'AngularRate_y':float, 'AngularRate_z':float,
                'Temperture':float, 'Pressure':float, 'MagnetCount':int, 'MagnetSwitch':int,
                },
            #usecols=lambda x: x in direct_acc+[index_col],
            usecols=lambda x: x in direct_acc+['time'],
            )
           #).compute()

    @staticmethod
    def sample_data(buf, data_read_by_api, data_sampled_first, data_sampled_last):
        '加速度データファイルの必要部分(data_sampled_firstからdata_sampled_last)を抽出する'
        # 加速度データファイルの不要部分(上3行)を削除.
        cmd1 = "sed -e 1,3d {data_read_by_api}".format(data_read_by_api=data_read_by_api)
        # 加速度データファイルの必要部分(data_sampled_firstからdata_sampled_last)を抽出.
        cmd2 = "sed -n {start},{end}p".format(start=data_sampled_first, end=data_sampled_last)
        res1 = sp.Popen(cmd1.split(" "), stdout=sp.PIPE)
        with open(buf, 'w') as f:
            sp.Popen(cmd2.split(" "), stdin=res1.stdout, stdout=f)
        #sp.run(['awk', '-F', '","', '{print}'])


##################################################################
##########データフレーム型変数をプロットするクラス################
##################################################################
class DataframePlotter():
    'DataFrameMakerクラスから生成したDataFrame型変数をプロットする'
    'メソッドはクラスの継承で書くべき！！'
    '親クラス；plot、子クラス；time_pred_plot、acc1_acc2_plot'
    'ToDoリストの書き方：リーダブルコード 5.2 自分の考えを記録する'
    def __init__(self, df, delta):
        self.df = df  # 加速度データを平均化
        self.delta = int(delta/config.mean_range)    # 平均値をとる要素数で区間を割る

    @staticmethod
    def plot(df, delta, args):  # delta:グラフの定義域,*args:グラフを描く列のタプル(＊タプルで受け取る)
        'DataFrame型変数をプロットする'
        pass


class TimePredDataframePlotter(DataframePlotter):

    def plot(self, args):
        '加速度・角速度の時系列変化をプロット'
        predict = pd.DataFrame(config.pred_by_prob_model, columns=['pred'])
        self.df = pd.concat([(self.df)[list(args)], predict], axis=1)
        for i in range(int(len(self.df)/(self.delta))):
            copy_df = (self.df).loc[(self.delta)*i:(self.delta)*(i+1), :]
            copy_df.dropna(how='all')
            ax1 = copy_df[list(args)].plot()
            ax = copy_df[['pred']].plot(ax=ax1)
            ax.set_title(config.data_read_by_api)
            #ax.set_ylim([-5.0, 2.5])
            plt.show()
            #plt.savefig(os.save_graph_to_path.join(PATH, "demo"+str(i)+".png"))
            ## テスト用グラフの保存先
            plt.savefig(os.path.join('../tests/test_plot/', "demo"+str(i)+".png"))


class Acc1Acc2DataframePlotter(DataframePlotter):

    def plot(self):
        '加速度の2次元データをプロットする'
        #ax = (self.df).plot.scatter(x=config.direct_acc[0], y=config.direct_acc[1])   # 散布図
        #ax = (self.df).plot.scatter(x=config.data_sampled_by_func['Acceleration_x'], y=config.data_sampled_by_func['Acceleration_y'])   # 散布図
        ax = (self.df).plot.scatter(x='Acceleration_x', y='Acceleration_y')   # 散布図
        ax.set_title(config.data_read_by_api)
        ax.set_xlim([-1.5, 0.5])
        ax.set_ylim([-2.0, 0.5])
        plt.show()
        # FIXME:テスト用の設定は、テストコードに書くべき！
        ## テスト用グラフの保存先
        #plt.savefig(os.path.join('../tests/test_plot/', "demo"+".png"))


class Acc1Acc2ColorDataframePlotter(DataframePlotter):

    def plot(self, args):
        '加速度の2次元データをプロットする'
        # 1.予測値を結合
        predict = pd.DataFrame(config.pred_by_prob_model, columns=['pred'])
        self.df = pd.concat([(self.df)[list(args)], predict], axis=1)
        # 2.散布図をプロット
        #ax = (self.df).plot.scatter(x=config.direct_acc[0], y=config.direct_acc[1])   # 散布図
        #ax = (self.df).plot.scatter(x=config.data_sampled_by_func['Acceleration_x'], y=config.data_sampled_by_func['Acceleration_y'])   # 散布図
        ax = (self.df).plot.scatter(x='Acceleration_x', y='Acceleration_y', vmin=0, vmax=2, c=(self.df).pred, cmap=cm.rainbow)   # 散布図
        ax.set_title(config.data_read_by_api)
        ax.set_xlim([-1.5, 0.5])
        ax.set_ylim([-2.0, 2.0])
        #plt.colorbar(ax)
        plt.show()
        # FIXME:テスト用の設定は、テストコードに書くべき！
        ## テスト用グラフの保存先
        #plt.savefig(os.path.join('../tests/test_plot/', "demo"+".png"))


##################################################################
##########メイン関数##############################################
##################################################################
def main():
    '確率モデルを適用し、学習結果を時系列表示する'
    'もしくは、加速度データを2次元プロットする'

    'グローバル変数のセット'
    #global_parameter = Global(
    #    input_pred_by_prob_model=None,
    #    input_data_read_by_api="../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv",  # ID16
    #    #input_data_read_by_api="../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv",  # ID19
    #    input_direct_acc=[
    #        'Acceleration_x',
    #        'Acceleration_y',
    #        'Acceleration_z',
    #        #'AngularRate_x',
    #        #'AngularRate_y',
    #        #'AngularRate_z',
    #        ],
    #    #input_save_graph_to_path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100",
    #    input_save_graph_to_path="/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hoge-hoge",
    #    #input_save_graph_to_path="/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100",
    #    #input_save_graph_to_path="/Users/okimototakuya/Desktop/tmp",
    #    input_plot_amount_in_graph=10000,
    #    #input_plot_amount_in_graph=131663,
    #    input_data_sampled_first=60000,
    #    input_data_sampled_last=69999,
    #    input_data_sampled_by_func=None,
    #    )

    '加速度データのDataFrame型変数を属性とする、DataframeMaker型オブジェクトを作成'
    buf = "../dataset/buf.csv"  # 加工したcsvファイルの保存先
    DataframeMaker.sample_data(buf, config.data_read_by_api, config.data_sampled_first, config.data_sampled_last)
    #data_sampled_by_func = DataframeMaker(buf, config.direct_acc, config.data_sampled_first, config.data_sampled_last)
    data_sampled_by_func = DataframeMaker(buf, config.direct_acc, config.data_sampled_first, config.data_sampled_last)

    'メインプログラム実行時の引数によって、描画するグラフを決定&プロット'
    try:
        if sys.argv[1] in ['0', '1', '2', '3']:
            if sys.argv[1] == '0':    # 隠れマルコフモデル
                #np.set_printoptions(threshold=np.inf)    # 配列の要素を全て表示(状態系列)
                hmm_learn.hmmLearn(data_sampled_by_func.df)
                #pred = hmm_learn.pred
                TimePredDataframePlotter(config.data_sampled_by_func, config.plot_amount_in_graph).plot(tuple(config.direct_acc))
            elif sys.argv[1] == '1':    # クラスタリング
                #np.set_printoptions(threshold=np.inf)    # 配列の要素を全て表示(状態系列)
                cluster_learn.clusterLearn(data_sampled_by_func.df)
                #pred = cluster_learn.pred
                TimePredDataframePlotter(config.data_sampled_by_func, config.plot_amount_in_graph).plot(tuple(config.direct_acc))
            elif sys.argv[1] == '2':    # 加速度を２次元プロット
                config.data_sampled_by_func = config.aveData(data_sampled_by_func.df)
                Acc1Acc2DataframePlotter(config.data_sampled_by_func, config.plot_amount_in_graph).plot()
            elif sys.argv[1] == '3':    # 加速度を２次元プロット(予測値による色付き)
                #np.set_printoptions(threshold=np.inf)    # 配列の要素を全て表示(状態系列)
                hmm_learn.hmmLearn(data_sampled_by_func.df)
                #pred = hmm_learn.pred
                #FIXME:クラス名を変更する必要がある.
                Acc1Acc2ColorDataframePlotter(config.data_sampled_by_func, config.plot_amount_in_graph).plot(tuple(config.direct_acc))
        else:
            raise WrongArgumentException(sys.argv[1])
    except IndexError as err:
        print("Pythonスクリプトに引数が与えられていません.(sys.argv[1]:0~2)")
        sys.exit()
    #except Exception as other:
    #    print("予期せぬエラーです.")
    #    sys.exit()

    #'グラフを描画'
    #DataframePlotter.plot(data_sampled_by_func.df, config.plot_amount_in_graph, tuple(config.direct_acc))

if __name__ == '__main__':
    main()
