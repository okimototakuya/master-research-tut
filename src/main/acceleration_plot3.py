import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('../test')
import test_acceleration_plot3 as tap3
#import hmmlearn
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# 2021/7/1:HACK1: Pythonにおけるグローバル変数の取り扱いについて
# 方法1.python公式ドキュメント(https://docs.python.org/ja/3/faq/programming.html):グローバル変数モジュールのグローバル変数はカプセル化せず、剥き出し.
# 方法2.実践Python3:シングルトンデザインパターンでは、変数をプライベート化し、変数を取得するメソッドをパブリック化.
# 2021/7/1:HACK2: PLOT_AMOUNT_IN_GRAPHの必要性について。一応config.pyから移動させてけども。

# 加速度データファイル(csv)のパス
#PATH_CSV_ACCELERATION_DATA = "../../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16
#PATH_CSV_ACCELERATION_DATA = "../../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19
#PATH_CSV_ACCELERATION_DATA = "../../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16(交差点ラベル付)
#PATH_CSV_ACCELERATION_DATA = "../../dataset/labeledEditedLOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19(交差点ラベル付)
PATH_CSV_ACCELERATION_DATA = "../../dataset/83番交差点.csv"

# 時系列/加速度2次元プロット画像ファイルの保存先
#PATH_PNG_PLOT_DATA = "/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100/"
PATH_PNG_PLOT_DATA = "/Users/okimototakuya/Desktop/研究データ/サンプル2件/ID16/hoge-hoge/"
#PATH_PNG_PLOT_DATA = "/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100/"
#PATH_PNG_PLOT_DATA = "/Users/okimototakuya/Desktop/tmp/"

# csvファイルを読み取る際の、切り出し区間
DATA_SAMPLED_FIRST = 0  # 切り出し始め(line値DATA_SAMPLED_FIRSTはDataFrame型変数に含まれる)
#DATA_SAMPLED_LAST = 1000 # 切り出し終わり(line値DATA_SAMPLED_LASTはDataFrame型変数に含まれない)
#DATA_SAMPLED_LAST = sum([1 for _ in open(PATH_CSV_ACCELERATION_DATA)]) - 1  # 最後のサンプル
DATA_SAMPLED_LAST = 30 # テスト用

# 平均値計算の設定: 関数average_data
MEAN_RANGE = 5  # 平均値を計算する際の、要素数
HOW_TO_CALCULATE_MEAN = 'fixed_mean'    # 平均値の算出方法 ('fixed_mean': 固定(?)平均, 'slide_mean': 移動平均, 'slide_median': 移動中央値)

# 確率モデルの設定: 関数estimate_state_data
ASSUMED_PROBABILISTIC_MODEL = 'hmm' # 仮定する確率モデル (クラスタリング: 'clustering', 隠れマルコフモデル: 'hmm')
NUMBER_OF_ASSUMED_STATE = 3 # 仮定する状態数(クラスタ数)

# プロットの設定: 関数plot_data
#PLOT_AMOUNT_IN_GRAPH = 10000   # 1つのグラフにおけるプロット数
#PLOT_AMOUNT_IN_GRAPH = 131663
HOW_TO_PLOT = 'pd' # プロットに用いるライブラリ (pd.DataFrame.plot: 'pd', seaborn.pairplot: 'sns')

def read_csv_(input_path_to_csv):
    '''
    csvファイル(加速度データ)を読み込み、pd.DataFrame型変数を返す関数
    引数1:csvファイルの相対パス
    '''
    default_num_skip_row = 1    # 列名の行に欠損値 (None) が含まれるため、スキップし、列名をユーザが再定義(names)
    return pd.read_csv(
            input_path_to_csv,  # 入力のcsvファイルパス
            index_col = 0,  # 列0 (列名オブジェクトがNone) をインデックスに
                    # 切り出し始め(line値DATA_SAMPLED_FIRSTはDataFrame型変数に含まれる)
            skipfooter = sum([1 for _ in open(input_path_to_csv)]) - (DATA_SAMPLED_LAST + default_num_skip_row),    \
                    # 切り出し終わり(line値DATA_SAMPLED_LASTはDataFrame型変数に含まれない)
            header = None,
            names = ['Unnamed: 0', 'line', 'time',
                'Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]',
                'AngularRate(X)[dps]', 'AngularRate(Y)[dps]', 'AngularRate(Z)[dps]',
                'Temperature[degree]', 'Pressure[hPa]', 'MagnetCount', 'MagnetSwitch',
                'onCrossroad', 'crossroadID'],
            skiprows = DATA_SAMPLED_FIRST + default_num_skip_row,
            engine = 'python',
            )


def average_data(input_acc_ang_df, input_mean_range, input_how):
    '''
    pd.DataFrame型変数を引数にし、特定区間における平均値を算出し、pd.DataFrame型変数を返す関数
    引数1:pd.DataFrame型変数の加速度/角速度の列(→pd.DataFrame型)
    引数2:平均値を計算する際の、要素数
    引数3:平均値の算出方法 fixed_mean:固定(?)平均, slide_mean:移動平均, slide_median:移動中央値'
    '''
    if input_how == 'fixed_mean':  # 固定(?)平均
       #len_after_division = int(len(input_acc_ang_df)/input_mean_range)    # 固定平均を算出した際、算出後のpd.DataFrame型変数の大きさ
       return pd.concat([(input_acc_ang_df.iloc[offset_i:offset_i+input_mean_range].describe()).loc['mean', :] \
               #for offset_i in range(0, len_after_division, input_mean_range)], axis=1).T.reset_index(drop='index') # インデックスオブジェクトの型はpd.Int64Index (pd.read_csvのデフォルト)
               for offset_i in range(0, len(input_acc_ang_df), input_mean_range)], axis=1).T.reset_index(drop='index')  \
                       .join(input_acc_ang_df['time'][::input_mean_range].reset_index(drop='index'))
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
    '''
    主成分分析を実行する関数
    '''
    # 行列の標準化(各列に対して、平均値を引いたものを標準偏差で割る)
    df_averaged_std = input_df_averaged.iloc[:, 0:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    # 主成分分析の実行
    pca = PCA()
    pca.fit(df_averaged_std)
    # データを主成分空間に写像
    ndarray_feature = pca.transform(df_averaged_std)
    # 主成分得点
    df_pca = pd.DataFrame(ndarray_feature, columns=["PC{}".format(x+1) for x in range(len(df_averaged_std.columns))])
    ## 第１主成分と第２主成分でプロット
    #plt.figure(figsize=(6, 6))
    #plt.scatter(ndarray_feature[:, 0], ndarray_feature[:, 1], alpha=0.8, c=list(df_averaged_std.iloc[:, 0]))
    #plt.grid()
    #plt.xlabel("PC1")
    #plt.ylabel("PC2")
    # 第１~６主成分まで、sns.pairplotでプロット
    #sns.pairplot(
    #        df_pca,
    #        diag_kind = 'kde',
    #        plot_kws = {'alpha': 0.2},
    #    )
    return df_pca


def estimate_state_data(input_df_averaged, input_how, input_number_of_assumed_state):
    '''
    隠れマルコフモデルを仮定し、pd.DataFrame型引数の訓練及び状態推定を行う関数
    '''
    if input_how == 'clustering':
        model = KMeans(n_clusters = input_number_of_assumed_state)   # クラスタリング(混合ガウス分布)の仮定
        model.fit(input_df_averaged)    # クラスタリングにより、引数のデータを訓練
        return model.labels_
    elif input_how == 'hmm':
        #model = hmmlearn.hmm.GaussianHMM(n_components=input_number_of_assumed_state, covariance_type="full")    # 隠れマルコフモデルの仮定
        model = hmm.GaussianHMM(n_components=input_number_of_assumed_state, covariance_type="full")    # 隠れマルコフモデルの仮定
        model.fit(input_df_averaged)    # 隠れマルコフモデルにより、引数のデータを訓練
        #np.set_printoptions(threshold=np.inf)  # 配列の要素を全て表示(状態系列)
        #print("初期確率\n", model.startprob_)
        #print("平均値\n", model.means_)
        #print("共分散値\n", model.covars_)
        print("遷移確率\n", model.transmat_)
        #print("対数尤度\n", model.score(input_df_averaged))
        #print("状態系列の復号\n", model.predict(input_df_averaged))
        return model.predict(input_df_averaged)
    else:
        raise Exception('input_howに無効な値{wrong_input_how}が与えられています.'.format(wrong_input_how=input_how))


def plot_data(input_df_averaged, input_ndarray_predicted, input_how):
    '''
    pd.DataFrame型変数のプロットを行う関数
    '''
    #4-1. pd.DataFrame.plotを用いて、プロットする場合
    if input_how == 'pd':
        input_df_averaged.plot(
                x = 'time',                             # 時系列プロット
                #x = input_df_averaged.columns[0],      # 特徴量/主成分の散布図プロット
                #x = input_df_averaged.columns[1],
                #x = input_df_averaged.columns[2],
                #x = input_df_averaged.columns[3],
                #x = input_df_averaged.columns[4],
                #x = input_df_averaged.columns[5],
                #y = input_df_averaged.columns[0],
                #y = input_df_averaged.columns[1],
                y = input_df_averaged.columns[2],
                #y = input_df_averaged.columns[3],
                #y = input_df_averaged.columns[4],
                #y = input_df_averaged.columns[5],
                #kind = 'scatter',
                #c = 'r',
                #c = input_ndarray_predicted,
                #cmap = 'rainbow'
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
    if (DATA_SAMPLED_FIRST >= DATA_SAMPLED_LAST) or (DATA_SAMPLED_FIRST < 0 or DATA_SAMPLED_LAST < 0):
        raise Exception('csvファイルの切り出し区間の指定が不適切です:{wrong_first}, {wrong_last}'.format(wrong_first=DATA_SAMPLED_FIRST, wrong_last=DATA_SAMPLED_LAST))
    else:
        # 1. csvファイル(加速度データ)を読み込み、pd.DataFrame型変数(df_read)を返す
        df_read = read_csv_(PATH_CSV_ACCELERATION_DATA)
        #df_read = df_read['onCrossroad']    # テスト: 列'onCrossroad'の抽出 (成功)
        #df_read = df_read[df_read['onCrossroad']=='0']    # 全ての交差点を抽出
        #df_read = df_read[df_read['crossroadID']=='83']    # 交差点83を抽出
        # 2. 上記で返されたdf_readについて、平均値を計算する(df_averaged)
        df_averaged = average_data(
                            input_acc_ang_df =  # 引数1:pd.DataFrame型変数の加速度/角速度の列(→pd.DataFrame型)
                                    df_read.loc[:,[  # 行数(データ数)の指定
                                        'time',                 # 時刻
                                        'Acceleration(X)[g]',   # 列(特徴量)の指定
                                        'Acceleration(Y)[g]',
                                        'Acceleration(Z)[g]',
                                        'AngularRate(X)[dps]',
                                        'AngularRate(Y)[dps]',
                                        'AngularRate(Z)[dps]',
                                       ]],
                            input_mean_range = MEAN_RANGE, # 引数2:平均値を計算する際の、要素数
                            input_how = HOW_TO_CALCULATE_MEAN,   # 引数3:平均値の算出方法 fixed_mean:固定(?)平均, slide_mean:移動平均, slide_median:移動中央値
                    )
        # 3. 主成分分析を実行する
        # FIXME2021/7/4: 上記の場合(main関数定義文下のif分岐)以外でも、切り出し区間によっては、関数decompose_dataで例外が発生する。
        # 例. (DATA_SAMPLED_FIRST, DATA_SAMPLED_LAST)=(5, 9)の時、ValueError: Shape of passed values is (4, 4), indices imply (4, 5)
        df_pca = decompose_data(df_averaged)
        # 4. 隠れマルコフモデルを適用する
        if NUMBER_OF_ASSUMED_STATE > (DATA_SAMPLED_LAST - DATA_SAMPLED_FIRST):  # 2021/7/5 2時頃: clustering, hmm共に、全く同じ例外が投げられることを確認した。
            raise Exception('確率モデルを用いる際に仮定する状態数の値が不適切です:(状態数, サンプル数)=({wrong_number_state}, {wrong_number_sample})'  \
                    .format(wrong_number_state=NUMBER_OF_ASSUMED_STATE, wrong_number_sample=DATA_SAMPLED_LAST-DATA_SAMPLED_FIRST))
        elif NUMBER_OF_ASSUMED_STATE == (DATA_SAMPLED_LAST - DATA_SAMPLED_FIRST)    \
                and ASSUMED_PROBABILISTIC_MODEL == 'hmm':   # HMMを仮定した場合、状態数=サンプル数の時でも警告や例外が発生する。
            # HACK2021/7/5: 今後、状態数とサンプル数、尤度関数の関係について考える機会があると思う。
            # [例外パターン]: (DATA_SAMPLED_FIRST, DATA_SAMPLED_LAST)=(0, 3), NUMBER_OF_ASSUMED_STATE=3の時、ValueError: rows of transmat_ must sum to 1.0 (got [1. 0. 1.])
            # [警告パターン]: (DATA_SAMPLED_FIRST, DATA_SAMPLED_LAST)=(108, 111), NUMBER_OF_ASSUMED_STATE=3の時、
            #   Fitting a model with 89 free scalar parameters with only 18 data points will result in a degenerate solution.
            #   /Users/okimototakuya/anaconda3/lib/python3.6/site-packages/seaborn/distributions.py:306: UserWarning: Dataset has 0 variance; skipping density estimate.
            #   warnings.warn(msg, UserWarning)
            #   /Users/okimototakuya/anaconda3/lib/python3.6/site-packages/seaborn/distributions.py:306: UserWarning: Dataset has 0 variance; skipping density estimate.
            #   warnings.warn(msg, UserWarning)
            #   /Users/okimototakuya/anaconda3/lib/python3.6/site-packages/seaborn/distributions.py:306: UserWarning: Dataset has 0 variance; skipping density estimate.
            #   warnings.warn(msg, UserWarning), NUMBER_OF_ASSUMED_STATE=3の時、ValueError: rows of transmat_ must sum to 1.0 (got [1. 0. 1.])
            raise Exception('HMMを仮定した場合、状態数=サンプル数の時でも警告や例外が発生します:(状態数, サンプル数)=({wrong_number_state}, {wrong_number_sample})' \
                    .format(wrong_number_state=NUMBER_OF_ASSUMED_STATE, wrong_number_sample=DATA_SAMPLED_LAST-DATA_SAMPLED_FIRST))
        else:
            ndarray_predicted_original = estimate_state_data(   # 主成分分析をせずに、隠れマルコフモデルを適用する場合
                                    input_df_averaged = df_averaged,
                                    input_how = ASSUMED_PROBABILISTIC_MODEL,
                                    input_number_of_assumed_state = NUMBER_OF_ASSUMED_STATE,
                                )
            ndarray_predicted_pca = estimate_state_data(   # 主成分分析をして、隠れマルコフモデルを適用する場合
                                    input_df_averaged = df_pca,
                                    input_how = ASSUMED_PROBABILISTIC_MODEL,
                                    input_number_of_assumed_state = NUMBER_OF_ASSUMED_STATE,
                                )
        # 5. 上記の算出結果をプロットする
        # 5-1. pd.DataFrame.plotを用いて、プロットする場合: input_how="pd"
        # 5-2. seaborn.pairplotを用いて、プロットする場合: input_how="sns"
        plot_data(  # 主成分分析をしなかったもの
                input_df_averaged = df_averaged,
                input_ndarray_predicted = ndarray_predicted_original,
                input_how = HOW_TO_PLOT,
            )
        plot_data(  # 主成分分析をしたもの
                input_df_averaged = df_pca,
                input_ndarray_predicted = ndarray_predicted_pca,
                input_how = HOW_TO_PLOT,
            )
        # プロットの可視化
        # IPython環境でなくターミナル環境で実行する場合、プロットを可視化するのに必須
        # [関連]: decompose_data, plot_data
        plt.show()

if __name__ == '__main__':
    main()
