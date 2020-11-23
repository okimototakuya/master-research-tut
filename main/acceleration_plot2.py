import os
import sys
import time
import subprocess as sp
import pandas as pd
#import dask.dataframe as dd
import matplotlib
#matplotlib.use('Agg')  # pyplotで生成した画像を保存するためのインポート
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import config
import hmm_learn
import cluster_learn


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

    @classmethod
    def sample_data(cls, buf, data_read_by_api, data_sampled_first, data_sampled_last):
        '加速度データファイルの必要部分(data_sampled_firstからdata_sampled_last)を抽出する'
        '抽出した加速度データファイルはbufディレクトリ下に保存する'
        # 加速度データファイルの不要部分(上3行)を削除.
        cmd1 = "sed -e 1,3d {data_read_by_api}".format(data_read_by_api=data_read_by_api)
        # 加速度データファイルの必要部分(data_sampled_firstからdata_sampled_last)を抽出.
        cmd2 = "sed -n {start},{end}p".format(start=data_sampled_first, end=data_sampled_last)
        res1 = sp.Popen(cmd1.split(" "), stdout=sp.PIPE)
        with open(buf, 'w') as f:
            res2 = sp.Popen(cmd2.split(" "), stdin=res1.stdout, stdout=f)
        #sp.run(['awk', '-F', '","', '{print}'])
        while True:
            ret2 = res2.poll()
            if ret2 is None:
                print('waiting for finish')
                time.sleep(1)
            else:
                break
        while True:
            ret1 = res1.poll()
            if ret1 is None:
                print('waiting for finish')
                time.sleep(1)
            else:
                break
        res1.stdout.close()
        #res2.stdout.close()
        f.closed


##################################################################
##########データフレーム型変数をプロットするクラス################
##################################################################
class DataframePlotter():
    'DataFrameMakerクラスから生成したDataFrame型変数をプロットする'
    'メソッドはクラスの継承で書くべき！！'
    '親クラス；plot、子クラス；time_pred_plot、acc1_acc2_plot'
    'ToDoリストの書き方：リーダブルコード 5.2 自分の考えを記録する'
    def __init__(self, df, delta, input_state):
        self.df = df  # 加速度データを平均化
        self.delta = int(delta/config.mean_range)    # 平均値をとる要素数で区間を割る
        #self.delta = delta
        #self.__state = input_state
        self.state = input_state
        #self.generate_graph = None

        #if input_state == 'p':
        #    #self.generate_graph = plt.show
        #    self.generate_graph = lambda x:
        #elif input_state == 's':
        #    #self.generate_graph = plt.savefig
        #else:
        #    raise ValueError(input_state)

        self.__save_graph_to_path = None

    @property
    def save_graph_to_path(self):
        return self.__save_graph_to_path

    @save_graph_to_path.setter
    def save_graph_to_path(self, input_save_graph_to_path):
        self.__save_graph_to_path = input_save_graph_to_path

    #def generate_graph(self, input_save_graph_to_path):
    def generate_graph(self):
        'HACK'
        '185~191行のコードを抽象化した関数'
        '1.plt.show():引数を受け取れない→self.generate_graphでplt.savefigと同じ扱いをするのは不可能'
        '2.plt.savefig():TimePredDataframePlotterクラスに関しては、グラフの保存場所を更新する必要がある→抽象化する必要あり'
        #if input_state == 'p':
        if self.state == 'p':
        #if self.state == 'p' and input_save_graph_to_path == None:
            print('pに入ってます.')
            return plt.show()
        #elif input_state == 's':
        elif self.state == 's':
            #elif self.state == 's' and input_save_graph_to_path != None:   # テストコードから：条件入らない→else:「入ってないです.」
            print('sに入ってます.')
            #return plt.savefig(self.__save_graph_to_path)
            #return plt.savefig(input_save_graph_to_path)
            return plt.savefig(self.save_graph_to_path + 'demo.png')
            #return print('aaa')
        else:
            print('入ってないです.')

    def plot(self, df, delta, args):  # delta:グラフの定義域,*args:グラフを描く列のタプル(＊タプルで受け取る)
        'DataFrame型変数をプロットする'
        pass

    '明示的に関数定義を書く必要ある？'
    'ない(参考：multiplexer2.py→セッターメソッドはインスタンス生成と同等のイメージ)'
    'インスタンス変数生成時、self.~って形で宣言(？)する必要あり.'
    #def generate_graph(self):
    #    pass


class TimePredDataframePlotter(DataframePlotter):

    def plot(self):
        '加速度・角速度の時系列変化をプロット'
        if sys.argv[1] != '0':    # 本モジュールに引数0を渡して実行した場合のみ、予測値なし時系列グラフをプロットする
            predict = pd.DataFrame(config.pred_by_prob_model, columns=['pred'])
            self.df = pd.concat([(self.df)[config.direct_acc], predict], axis=1)
        for i in range(int(len(self.df)/(self.delta))):
            copy_df = (self.df).loc[(self.delta)*i:(self.delta)*(i+1), :]
            copy_df.dropna(how='all')
            ax1 = copy_df[config.direct_acc].plot()
            if sys.argv[1] != '0':    # 本モジュールに引数0を渡して実行した場合のみ、予測値なし時系列グラフをプロットする
                ax = copy_df[['pred']].plot(ax=ax1)
                ax.set_title(config.data_read_by_api)
            ax1.set_title(config.data_read_by_api)
            #ax.set_ylim([-5.0, 2.5])
            #plt.show()
            #plt.savefig(os.path.join(PATH, "demo"+str(i)+".png"))
            ## テスト用グラフの保存先
            #plt.savefig(os.path.join('../tests/test_plot/', "demo"+str(i)+".png"))
            #self.generate_graph(self.save_graph_to_path)
            self.generate_graph()


class Acc1Acc2DataframePlotter(DataframePlotter):

    def plot(self):
        '加速度の2次元データをプロットする'
        #ax = (self.df).plot.scatter(x=config.direct_acc[0], y=config.direct_acc[1])   # 散布図
        #ax = (self.df).plot.scatter(x=config.data_sampled_by_func['Acceleration_x'], y=config.data_sampled_by_func['Acceleration_y'])   # 散布図
        #ax = (self.df).plot.scatter(x='Acceleration_x', y='Acceleration_y')   # 散布図
        ax = (self.df).plot.scatter(x=config.direct_acc[0], y=config.direct_acc[1])   # 散布図
        ax.set_title(config.data_read_by_api)
        ax.set_xlim([-1.5, 0.5])
        ax.set_ylim([-2.0, 0.5])
        #plt.show()
        # FIXME:テスト用の設定は、テストコードに書くべき！
        ## テスト用グラフの保存先
        #plt.savefig(os.path.join('../tests/test_plot/', "demo"+".png"))
        self.generate_graph()


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
        #plt.show()
        # FIXME:テスト用の設定は、テストコードに書くべき！
        ## テスト用グラフの保存先
        #plt.savefig(os.path.join('../tests/test_plot/', "demo"+".png"))
        self.generate_graph()


##################################################################
##########メイン関数##############################################
##################################################################
def main():
    '確率モデルを適用し、学習結果を時系列表示する'
    'もしくは、加速度データを2次元プロットする'

    '加速度データのDataFrame型変数を属性とする、DataframeMaker型オブジェクトを作成'
    buf = "../dataset/buf.csv"  # 加工したcsvファイルの保存先
    DataframeMaker.sample_data(buf, config.data_read_by_api, config.data_sampled_first, config.data_sampled_last)
    #data_sampled_by_func = DataframeMaker(buf, config.direct_acc, config.data_sampled_first, config.data_sampled_last)
    data_sampled_by_func = DataframeMaker(buf, config.direct_acc, config.data_sampled_first, config.data_sampled_last)

    'メインプログラム実行時の引数によって、描画するグラフを決定&プロット'
    'HACK:インジェクション攻撃に注意(参考:実践Python3 Interpreterパターン)'
    state_plot = 'p'    # 'p':plt.show()/'s':plt.savefig(config.save_graph_to_path)
    try:
        if sys.argv[1] in ['0', '1', '2', '3', '4']:
            if sys.argv[1] == '0':    # 時系列プロット：予測値なし
                #np.set_printoptions(threshold=np.inf)    # 配列の要素を全て表示(状態系列)
                data_sampled_by_func.df = config.aveData(data_sampled_by_func.df)
                tpdfp = TimePredDataframePlotter(data_sampled_by_func.df, config.plot_amount_in_graph, state_plot)
                if state_plot == 's':
                    tpdfp.save_graph_to_path = config.save_graph_to_path
                tpdfp.plot()
            elif sys.argv[1] == '1':    # 時系列プロット：隠れマルコフモデル
                #np.set_printoptions(threshold=np.inf)    # 配列の要素を全て表示(状態系列)
                hmm_learn.hmmLearn(data_sampled_by_func.df)
                #pred = hmm_learn.pred
                tpdfp = TimePredDataframePlotter(config.data_sampled_by_func, config.plot_amount_in_graph, state_plot)
                if state_plot == 's':
                    tpdfp.save_graph_to_path = config.save_graph_to_path
                tpdfp.plot()
            elif sys.argv[1] == '2':    # 時系列プロット：クラスタリング
                #np.set_printoptions(threshold=np.inf)    # 配列の要素を全て表示(状態系列)
                cluster_learn.clusterLearn(data_sampled_by_func.df)
                #pred = cluster_learn.pred
                TimePredDataframePlotter(config.data_sampled_by_func, config.plot_amount_in_graph, 'p').plot(tuple(config.direct_acc))
            elif sys.argv[1] == '3':    # 加速度を２次元プロット
                config.data_sampled_by_func = config.aveData(data_sampled_by_func.df)
                aadfp = Acc1Acc2DataframePlotter(config.data_sampled_by_func, config.plot_amount_in_graph, state_plot)
                if state_plot == 's':
                    aadfp.save_graph_to_path = config.save_graph_to_path
                aadfp.plot()
            elif sys.argv[1] == '4':    # 加速度を２次元プロット(予測値による色付き)
                #np.set_printoptions(threshold=np.inf)    # 配列の要素を全て表示(状態系列)
                hmm_learn.hmmLearn(data_sampled_by_func.df)
                #pred = hmm_learn.pred
                #FIXME:クラス名を変更する必要がある.
                Acc1Acc2ColorDataframePlotter(config.data_sampled_by_func, config.plot_amount_in_graph, 'p').plot(tuple(config.direct_acc))
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
