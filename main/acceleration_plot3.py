import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('../tests')
import test_acceleration_plot3 as tap3
import config
#import hmmlearn
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


' 加速度データファイル(csv)のパス'
#PATH_CSV_ACCELERATION_DATA = "../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16
#PATH_CSV_ACCELERATION_DATA = "../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19
PATH_CSV_ACCELERATION_DATA = "../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16(交差点ラベル付)
#PATH_CSV_ACCELERATION_DATA = "../dataset/labeledEditedLOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19(交差点ラベル付)


' 時系列/加速度2次元プロット画像ファイルの保存先'
#PATH_PNG_PLOT_DATA = "/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100/"
PATH_PNG_PLOT_DATA = "/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hoge-hoge/"
#PATH_PNG_PLOT_DATA = "/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100/"
#PATH_PNG_PLOT_DATA = "/Users/okimototakuya/Desktop/tmp/"


def read_csv_(input_path_to_csv):
    'csvファイル(加速度データ)を読み込み、pd.DataFrame型変数を返す関数  \
     引数1:csvファイルの相対パス'
    default_num_skip_row = 1    # 列名の行に欠損値 (None) が含まれるため、スキップし、列名をユーザが再定義(names)
    return pd.read_csv(
            input_path_to_csv,  # 入力のcsvファイルパス
            index_col = 0,  # 列0 (列名オブジェクトがNone) をインデックスに
            skiprows = tap3.TEST_DATA_SAMPLED_FIRST + default_num_skip_row,    \
                    # テスト:切り出し始め(line値TEST_DATA_SAMPLED_FIRSTはDataFrame型変数に含まれる)
            skipfooter = sum([1 for _ in open(input_path_to_csv)]) - (tap3.TEST_DATA_SAMPLED_LAST + default_num_skip_row),    \
                    # テスト:切り出し終わり(line値TEST_DATA_SAMPLED_LASTはDataFrame型変数に含まれない)
            #skiprows = config.data_sampled_first + default_num_skip_row,    \
            #        # 切り出し始め(line値config.data_sampled_firstはDataFrame型変数に含まれる)
            #skipfooter = sum([1 for _ in open(input_path_to_csv)]) - (config.data_sampled_last + default_num_skip_row),    \
            #        # 切り出し終わり(line値config.data_sampled_lastはDataFrame型変数に含まれない)
            header = None,
            names = ['Unnamed: 0', 'line', 'time',
                'Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]',
                'AngularRate(X)[dps]', 'AngularRate(Y)[dps]', 'AngularRate(Z)[dps]',
                'Temperature[degree]', 'Pressure[hPa]', 'MagnetCount', 'MagnetSwitch',
                'onCrossroad', 'crossroadID'],
            engine = 'python',
            )


def average_data(input_acc_ang_df, input_mean_range, input_how):
    'pd.DataFrame型変数を引数にし、特定区間における平均値を算出し、pd.DataFrame型変数を返す関数 \
     引数1:pd.DataFrame型変数の加速度/角速度の列(→pd.DataFrame型)    \
     引数2:平均値を計算する際の、要素数 \
     引数3:平均値の算出方法 fixed_mean:固定(?)平均, slide_mean:移動平均, slide_median:移動中央値'
    if input_how == 'fixed_mean':  # 固定(?)平均
       #len_after_division = int(len(input_acc_ang_df)/input_mean_range)    # 固定平均を算出した際、算出後のpd.DataFrame型変数の大きさ
       return pd.concat([(input_acc_ang_df.iloc[offset_i:offset_i+input_mean_range].describe()).loc['mean', :] \
               #for offset_i in range(0, len_after_division, input_mean_range)], axis=1).T.reset_index(drop='index') # インデックスオブジェクトの型はpd.Int64Index (pd.read_csvのデフォルト)
               for offset_i in range(0, len(input_acc_ang_df), input_mean_range)], axis=1).T.reset_index(drop='index') # インデックスオブジェクトの型はpd.Int64Index (pd.read_csvのデフォルト)
    elif input_how == 'slide_mean': # 移動平均
       #len_after_division = int(len(input_acc_ang_df)/input_mean_range)    # 固定平均を算出した際、算出後のpd.DataFrame型変数の大きさ
       #len_after_division = 28
       return pd.concat([(input_acc_ang_df.iloc[offset_i:offset_i+input_mean_range].describe()).loc['mean', :] \
               for offset_i in range(len(input_acc_ang_df)-input_mean_range+1)], axis=1).T.reset_index(drop='index') # インデックスオブジェクトの型はpd.Int64Index (pd.read_csvのデフォルト)
    elif input_how == 'slide_median':   # 移動中央値
       return pd.concat([(input_acc_ang_df.iloc[offset_i:offset_i+input_mean_range].describe()).loc['50%', :] \
               for offset_i in range(len(input_acc_ang_df)-input_mean_range+1)], axis=1).T.reset_index(drop='index') # インデックスオブジェクトの型はpd.Int64Index (pd.read_csvのデフォルト)
    else:
        raise Exception('input_howに無効な値{wrong_input_how}が与えられています.'.format(wrong_input_how=input_how))


def decompose_data(input_df_averaged):
    '主成分分析を実行する'
    # 行列の標準化(各列に対して、平均値を引いたものを標準偏差で割る)
    df_averaged_std = input_df_averaged.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    # 主成分分析の実行
    pca = PCA()
    pca.fit(df_averaged_std)
    # データを主成分空間に写像
    ndarray_feature = pca.transform(df_averaged_std)
    # 主成分得点
    pd.DataFrame(ndarray_feature, columns=["PC{}".format(x+1) for x in range(len(df_averaged_std.columns))])
    # 第１主成分と第２主成分でプロット
    plt.figure(figsize=(6, 6))
    plt.scatter(ndarray_feature[:, 0], ndarray_feature[:, 1], alpha=0.8, c=list(df_averaged_std.iloc[:, 0]))
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")


def estimate_state_data(input_df_averaged, input_how):
    '隠れマルコフモデルを仮定し、pd.DataFrame型引数の訓練及び状態推定を行う関数'
    if input_how == 'clustering':
        model = KMeans(n_clusters = 3)   # クラスタリング(混合ガウス分布)の仮定
        model.fit(input_df_averaged)    # クラスタリングにより、引数のデータを訓練
        return model.labels_
    elif input_how == 'hmm':
        #model = hmmlearn.hmm.GaussianHMM(n_components=3, covariance_type="full")    # 隠れマルコフモデルの仮定
        model = hmm.GaussianHMM(n_components=3, covariance_type="full")    # 隠れマルコフモデルの仮定
        model.fit(input_df_averaged)    # 隠れマルコフモデルにより、引数のデータを訓練
        #np.set_printoptions(threshold=np.inf)  # 配列の要素を全て表示(状態系列)
        #print("初期確率\n", model.startprob_)
        #print("平均値\n", model.means_)
        #print("共分散値\n", model.covars_)
        #print("遷移確率\n", model.transmat_)
        #print("対数尤度\n", model.score(input_df_averaged))
        #print("状態系列の復号\n", model.predict(input_df_averaged))
        return model.predict(input_df_averaged)
    else:
        raise Exception('input_howに無効な値{wrong_input_how}が与えられています.'.format(wrong_input_how=input_how))


def plot_data(input_df_averaged, input_ndarray_predicted, input_how):
    '4. プロット'
    #4-1. pd.DataFrame.plotを用いて、プロットする場合
    if input_how == 'pd':
        input_df_averaged.plot(
                x = 'Acceleration(X)[g]',
                #x = 'Acceleration(Y)[g]',
                #x = 'Acceleration(Z)[g]',
                #x = 'AngularRate(X)[dps]',
                #X = 'AngularRate(Y)[dps]',
                #x = 'AngularRate(Z)[dps]',
                #y = 'Acceleration(X)[g]',
                y = 'Acceleration(Y)[g]',
                #y = 'Acceleration(Z)[g]',
                #y = 'AngularRate(X)[dps]',
                #X = 'AngularRate(Y)[dps]',
                #y = 'AngularRate(Z)[dps]',
                kind = 'scatter',
                #c = 'r',
                c = input_ndarray_predicted,
                cmap = 'rainbow'
               )
    #4-2. seaborn.pairplotを用いて、プロットする場合
    elif input_how == 'sns':
        ser_state = pd.Series(
                input_ndarray_predicted,
                name = 'state',
            )
        df_averaged_state = pd.concat(
                [input_df_averaged, ser_state],
                axis = 1,
            )
        sns.pairplot(
                df_averaged_state,
                diag_kind = 'kde',
                plot_kws = {'alpha': 0.2},
                hue = 'state',
                palette = 'rainbow',
            )
    else:
        raise Exception('input_howに無効な値{wrong_input_how}が与えられています.'.format(wrong_input_how=input_how))


def main():
    '1. csvファイル(加速度データ)を読み込み、pd.DataFrame型変数(df_read)を返す'
    df_read = read_csv_(PATH_CSV_ACCELERATION_DATA)
    '2. 上記で返されたdf_readについて、平均値を計算する(df_averaged)'
    df_averaged = average_data(
                        input_acc_ang_df =  # 引数1:pd.DataFrame型変数の加速度/角速度の列(→pd.DataFrame型)
                                df_read.loc[:,[  # 行数(データ数)の指定
                                   'Acceleration(X)[g]',   # 列(特徴量)の指定
                                   'Acceleration(Y)[g]',
                                   'Acceleration(Z)[g]',
                                   "AngularRate(X)[dps]",
                                   "AngularRate(Y)[dps]",
                                   "AngularRate(Z)[dps]",
                                   ]],
                        input_mean_range = 1, # 引数2:平均値を計算する際の、要素数
                        input_how = 'fixed_mean',   # 引数3:平均値の算出方法 fixed_mean:固定(?)平均, slide_mean:移動平均, slide_median:移動中央値'
                )
    '主成分分析を実行する'
    decompose_data(df_averaged)
    '3. 上記で算出したdf_averagedについて、隠れマルコフモデルを適用する'
    # FIXME2021/6/25: バグ発生の条件２つ
    # 1. 切り出し始め: サンプル数=3の時、ValueError: rows of transmat_ must sum to 1.0 (got [0. 1. 1.])
    # 2. 切り出し区間: サンプル数 >= クラスタ数でないといけない。
    ndarray_predicted = estimate_state_data(
                            input_df_averaged = df_averaged,
                            input_how = 'clustering',
                        )
    '4. プロット'
    # 4-1. pd.DataFrame.plotを用いて、プロットする場合: input_how="pd"
    # 4-2. seaborn.pairplotを用いて、プロットする場合: input_how="sns"
    plot_data(
            input_df_averaged = df_averaged,
            input_ndarray_predicted = ndarray_predicted,
            input_how = 'sns',
        )
    'プロットの可視化'
    # IPython環境でなくターミナル環境で実行する場合、プロットを可視化するのに必須
    # [関連]: decompose_data, plot_data
    plt.show()

if __name__ == '__main__':
    main()
