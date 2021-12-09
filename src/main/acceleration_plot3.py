import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')                 # TkAggバックエンド: 最大で表示された後、すぐに最小化。
matplotlib.use('Qt5Agg')                 # 注. スクリプト環境では、pyqt5が必須。
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sys.path.append('../test')
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# 2021/7/1:HACK1: Pythonにおけるグローバル変数の取り扱いについて
# 方法1.python公式ドキュメント(https://docs.python.org/ja/3/faq/programming.html):グローバル変数モジュールのグローバル変数はカプセル化せず、剥き出し.
# 方法2.実践Python3:シングルトンデザインパターンでは、変数をプライベート化し、変数を取得するメソッドをパブリック化.

# 加速度データファイル(csv)のパス
#PATH_CSV_ACCELERATION_DATA = "../../dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16
#PATH_CSV_ACCELERATION_DATA = "../../dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19
#PATH_CSV_ACCELERATION_DATA = "../../dataset/labeledEditedLOG_20181219141837_00010533_0021002B401733434E45.csv"  # ID16(交差点ラベル付)
#PATH_CSV_ACCELERATION_DATA = "../../dataset/labeledEditedLOG_20181219141901_00007140_00140064401733434E45.csv"  # ID19(交差点ラベル付)
PATH_CSV_ACCELERATION_DATA = "../../dataset/32crossroad_50.csv"

# グラフ内のプロット量
AMOUNT_OF_PLOT = sum([1 for _ in open(PATH_CSV_ACCELERATION_DATA)]) - 1

# 平均値計算の設定: 関数average_data
MEAN_RANGE = 10  # 平均値を計算する際の、要素数
HOW_TO_CALCULATE_MEAN = 'slide_mean'    # 平均値の算出方法 ('fixed_mean': 固定(?)平均, 'slide_mean': 移動平均, 'slide_median': 移動中央値)

# 確率モデルの設定: 関数estimate_state_data
ASSUMED_PROBABILISTIC_MODEL = 'hmm' # 仮定する確率モデル (クラスタリング: 'clustering', 隠れマルコフモデル: 'hmm')
NUMBER_OF_ASSUMED_STATE = 3 # 仮定する状態数(クラスタ数)


def read_csv_(input_path_to_csv):
    '''
    csvファイル(加速度データ)を読み込み、pd.DataFrame型変数を返す関数
    引数1:csvファイルの相対パス
    '''
    default_num_skip_row = 1    # 列名の行に欠損値 (None) が含まれるため、スキップし、列名をユーザが再定義(names)
    return pd.read_csv(
            input_path_to_csv,  # 入力のcsvファイルパス
            index_col = 0,  # 列0 (列名オブジェクトがNone) をインデックスに
            header = None,
            names = ['Unnamed: 0', 'line', 'time',
                'Acceleration(X)[g]', 'Acceleration(Y)[g]', 'Acceleration(Z)[g]',
                'AngularRate(X)[dps]', 'AngularRate(Y)[dps]', 'AngularRate(Z)[dps]',
                'Temperature[degree]', 'Pressure[hPa]', 'MagnetCount', 'MagnetSwitch',
                'onCrossroad', 'crossroadID'],
            dtype = {'Unnamed: 0': int, 'line':int, 'time':"string",
                'Acceleration(X)[g]':float,  'Acceleration(Y)[g]':float,  'Acceleration(Z)[g]':float,
                'AngularRate(X)[dps]':float,  'AngularRate(Y)[dps]':float,  'AngularRate(Z)[dps]':float,
                'Temperature[degree]':float,  'Pressure[hPa]':float,  'MagnetCount':int, 'MagnetSwitch':int,
                'onCrossroad':int, 'crossroadID':int},
            skiprows = default_num_skip_row,
            engine = 'python',
            )


def average_data(input_acc_ang_df, input_mean_range, input_how):
    '''
    pd.DataFrame型変数を引数にし、特定区間における平均値を算出し、pd.DataFrame型変数を返す関数
    引数1:pd.DataFrame型変数の加速度/角速度の列(→pd.DataFrame型)
    引数2:平均値を計算する際の、要素数
    引数3:平均値の算出方法 fixed_mean:固定(?)平均, slide_mean:移動平均, slide_median:移動中央値'

    Parameters
    -----
    - input_acc_ang_df : pd.DataFrame (列 : 6つの加速度特徴量, 'time')
        ＊固定平均については、'time'列の更新作業が含まれるため。
    - input_mean_range : int
        平均区間
    - input_how : str
        平均値の算術方法

    Returns
    -----
    - pd.DataFrame
        平均値を格納した配列。ただし、'time'列は列尾に追加。
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
               #for offset_i in range(len(input_acc_ang_df)-input_mean_range+1)], axis=1).T.reset_index(drop='index')   # for文上と下: 下は元のinput_acc_ang_dfと大きさが変わらない。
               for offset_i in range(len(input_acc_ang_df))], axis=1).T.reset_index(drop='index')    \
                       .join(input_acc_ang_df['time'].reset_index(drop='index'))
    elif input_how == 'slide_median':   # 移動中央値
       return pd.concat([(input_acc_ang_df.iloc[offset_i:offset_i+input_mean_range].describe()).loc['50%', :] \
               #for offset_i in range(len(input_acc_ang_df)-input_mean_range+1)], axis=1).T.reset_index(drop='index') # for文上と下: 下は元のinput_acc_ang_dfと大きさが変わらない。
               for offset_i in range(len(input_acc_ang_df))], axis=1).T.reset_index(drop='index')    \
                       .join(input_acc_ang_df['time'].reset_index(drop='index'))
    else:
        raise Exception('input_howに無効な値{wrong_input_how}が与えられています.'.format(wrong_input_how=input_how))


def decompose_data(input_df_read):
    '''
    主成分分析を実行する関数
    '''
    # 行列の標準化(各列に対して、平均値を引いたものを標準偏差で割る)
    df_read_std = input_df_read.iloc[:, 0:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    # 主成分分析の実行
    pca = PCA()
    pca.fit(df_read_std)
    # データを主成分空間に写像
    ndarray_feature = pca.transform(df_read_std)
    # 主成分得点
    df_pca = pd.DataFrame(ndarray_feature, columns=["PC{}".format(x+1) for x in range(len(df_read_std.columns))])
    ## 第１主成分と第２主成分でプロット
    #plt.figure(figsize=(6, 6))
    #plt.scatter(ndarray_feature[:, 0], ndarray_feature[:, 1], alpha=0.8, c=list(df_read_std.iloc[:, 0]))
    #plt.grid()
    #plt.xlabel("PC1")
    #plt.ylabel("PC2")
    # 第１~６主成分まで、sns.pairplotでプロット
    #sns.pairplot(
    #        df_pca,
    #        diag_kind = 'kde',
    #        plot_kws = {'alpha': 0.2},
    #    )
    # ローディングの算出
    # pca.components_ : ノルム1の固有ベクトル
    # pca.explained_variance_ : 固有値
    loading = pca.components_ * np.c_[np.sqrt(pca.explained_variance_)]
    return df_pca, loading


def estimate_state_data(input_df_read, input_how='hmm', input_number_of_assumed_state=3):
    '''
    隠れマルコフモデルを仮定し、pd.DataFrame型引数の訓練及び状態推定を行う関数

    Parameters
    -----
    - input_df_read : pd.DataFrame
        入力配列。ただし、列'time'は含めない。
    - input_how : str
        確率モデルを表す文字列 ('clustering', 'hmm')
    - input_number_of_assumed_state : int
        仮定する状態数

    Returns
    -----
    - dict_param : dict
        推定した[パラメータ名, パラメータ値]を保持する辞書(dict)型
    '''
    if input_how == 'clustering':
        model = KMeans(n_clusters = input_number_of_assumed_state)   # クラスタリング(混合ガウス分布)の仮定
        model.fit(input_df_read)    # クラスタリングにより、引数のデータを訓練
        return model.labels_
    elif input_how == 'hmm':
        np.random.seed(seed=7)
        model = hmm.GaussianHMM(n_components=input_number_of_assumed_state, covariance_type="full", init_params='mtc')    # 隠れマルコフモデルの仮定
        model.startprob_ = np.array([1.0 if i == 0 else 0.0 for i in range(NUMBER_OF_ASSUMED_STATE)])   # 初期状態を状態1で固定
        model.fit(input_df_read)    # 隠れマルコフモデルにより、引数のデータを訓練
        #np.set_printoptions(threshold=np.inf)  # 配列の要素を全て表示(状態系列)
        #print("初期確率\n", model.startprob_)
        #print("平均値\n", model.means_)
        #print("共分散値\n", model.covars_)
        #print("遷移確率\n", model.transmat_)
        #print("対数尤度\n", model.score(input_df_read))
        #print("状態系列の復号\n", model.predict(input_df_read))
        np.set_printoptions(precision=3, suppress=True)        # 小数点以下の有効数字3桁, 指数表記しない
        dict_param = {
                 "初期確率": model.startprob_,
                 "平均値": model.means_,
                 "共分散値": model.covars_,
                 #"遷移行列": model.transmat_,  # デフォルト
                 "遷移行列": model.transmat_ * 100,   # 百分率
                 "対数尤度": model.score(input_df_read),
                 "状態系列の復号": model.predict(input_df_read)
                 }
        #return model.predict(input_df_read)
        return dict_param
    else:
        raise Exception('input_howに無効な値{wrong_input_how}が与えられています.'.format(wrong_input_how=input_how))


def plot_data(input_df_read, input_dict_param, input_loading=None):
    '''
    pd.DataFrame型変数のプロットを行う関数
    '''
    input_df_read = input_df_read.join(pd.Series(input_dict_param['状態系列の復号'], name='state')) # DataFrame配列と状態系列ndarray配列の結合
    # 4-0. プロットの保存先パスの設定
    str_path_to_crossroad = re.split('[/,\.]', PATH_CSV_ACCELERATION_DATA)[7]       # 交差点ラベルによるパス
    str_path_to_how_to_mean = HOW_TO_CALCULATE_MEAN + '_' + str(MEAN_RANGE)         # 平均方法及び平均区間によるパス
    # 4-1. 時系列プロット
    fig = plt.figure()
    fig.subplots_adjust(left=0.2)
    box_dic = {
            "facecolor" : "white",
            "edgecolor" : "darkblue",
            "boxstyle" : "Round",
            "linewidth" : 2
    }
    fig.text(0.01, 0.70, bbox=box_dic, s='- assumed state amount in HMM: {hmm}\n'\
                                         '- how to mean: {how}\n'\
                                         '- mean range: {range_}\n'\
                                         '- Factor Loading:\n{loading}\n'\
                                         '- transition matrix:\n{matrix}\n'\
                                         '- amount of plot: {amount}\n'\
                                         '- stay time in crossroad and around there:\n{stay}\n'\
                                         '- state series (first):\n{series_f}\n'\
                                         '- state series (last):\n{series_l}'\
                                         .format(hmm=NUMBER_OF_ASSUMED_STATE,
                                                 how=HOW_TO_CALCULATE_MEAN,
                                                 range_=MEAN_RANGE,
                                                 loading=input_loading,
                                                 matrix=input_dict_param['遷移行列'],
                                                 amount=AMOUNT_OF_PLOT,
                                                 stay=input_df_read['time'][AMOUNT_OF_PLOT-1]-input_df_read['time'][0],     # datetime型による演算: 日付計算
                                                 series_f=input_dict_param['状態系列の復号'][:25],
                                                 series_l=input_dict_param['状態系列の復号'][-25:]
                                                 )
            )
    fig.suptitle(PATH_CSV_ACCELERATION_DATA)
    mng = plt.get_current_fig_manager()                                                 # Mac環境で、pltによる自動フルスクリーンを用いる。
    mng.window.showMaximized()                                                          # QT (QtAgg5) バックエンド
    # 2021.12.4: 注: 'time'列の型をdatetime → object(str?) に変換
    # - pltのformatter/locatorを用いたプロットについて、datetime型の場合は他の型と異なる独自のformatter/locator型があるよう。例.DateFormatter
    # - DateFormatterでなく通常のFormatterを用いたところ、エラー/警告が発生することなく、ただ目盛り/ラベルが表示されないままプロットされた。
    input_df_read['time'] = input_df_read['time'].dt.strftime('%M:%S.%f')
    for i in range(1, 6+1):
        ax = fig.add_subplot(2, 3, i)
        ax = sns.scatterplot(              # 2021.11.17: HACK: seaborn.lineplot/scatterplotだと、plt.subplot使える。
                x = list(input_df_read.index),
                y = input_df_read.iloc[:, i-1],
                hue = input_df_read['state'],
                palette = 'rainbow'
            )
        # 4-1-2. Locatorの設定
        # - 目盛りの設定 (例. 線形目盛り, 対数目盛りなど)
        # - 下記
        # - xaxis: 線形目盛り
        # - yaxis: 自動目盛り (＊: 外れ値が含まれるため、ユーザが前もって目盛りの上限/下限を設定するのは望ましくない。
        #g.set_xticks(input_df_read['time'])                                        # Locatorは、FixedLocator: 2021.11.23: とりあえずのプロットに成功した。
        list_loc = list(input_df_read.index)
        ax.xaxis.set_major_locator(ticker.FixedLocator(list_loc[::10]))                                                # - 主目盛り
        ax.xaxis.set_minor_locator(ticker.FixedLocator(list(filter(lambda x: x % 10 != 0, list_loc))))                 # - 補助目盛り
        assert list_loc == sorted(list_loc[::10] + list(filter(lambda x: x % 10 != 0, list_loc)))                      # アサーション: 主目盛りと補助目盛りを足して、元のlist_locの要素を網羅するかどうか
        # 4-1-3. Formatterの設定
        # - 目盛りラベルの設定
        # - xticklabelsにリストを渡すと、その値の箇所だけ目盛りが配置される。
        # - ↑この時、FormatterはFixedFormatter
        xlabels_before_thinning_out = [input_df_read['time'][i].split('00000')[0] if i % 10 == 0 else '' for i in range(0, len(input_df_read))]  # 10点おきにx軸ラベルを表示. ただし、データそのものの間引きはなし.
        xlabels = list(filter(lambda x: x != '', xlabels_before_thinning_out))
        assert len(xlabels) == len(list_loc[::10])                  # アサーション: ラベルと主目盛りの個数が一致するかどうか。
        ax.set_xticklabels(labels=xlabels, rotation=90, fontsize=8)  # FormatterはFixedFormatter
        plt.grid(which='major')
    if input_loading is None:   # 元特徴量の場合、Figure1.pngとして保存
        #plt.savefig('../../plot/' + str_path_to_crossroad + '/' + str_path_to_how_to_mean + '/Figure1.png')
        plt.savefig('../../plot/' + 'hoge-hoge' + '/Figure1.png')                                               # テストプロット画像の保存先
    else:                       # PCA特徴量の場合、Figure3.pngとして保存
        #plt.savefig('../../plot/' + str_path_to_crossroad + '/' + str_path_to_how_to_mean + '/Figure3.png')
        plt.savefig('../../plot/' + 'hoge-hoge' + '/Figure3.png')                                               # テストプロット画像の保存先
    #4-2. 散布図プロット
    #plt.title(PATH_CSV_ACCELERATION_DATA)   # タイトル: この位置だと、時系列プロットの方に反映される。
    sns.pairplot(
            input_df_read,
            diag_kind = 'kde',
            plot_kws = {'alpha': 0.2},
            hue = 'state',
            palette = 'rainbow',
        )
    if input_loading is None:   # 元特徴量の場合、Figure1.pngとして保存
        #plt.savefig('../../plot/' + str_path_to_crossroad + '/' + str_path_to_how_to_mean + '/Figure2.png')
        plt.savefig('../../plot/' + 'hoge-hoge' + '/Figure2.png')                                               # テストプロット画像の保存先
    else:                       # PCA特徴量の場合、Figure3.pngとして保存
        #plt.savefig('../../plot/' + str_path_to_crossroad + '/' + str_path_to_how_to_mean + '/Figure4.png')
        plt.savefig('../../plot/' + 'hoge-hoge' + '/Figure4.png')                                               # テストプロット画像の保存先


def main():
    # 0. 標準入力がある場合、各パラメータを標準入力から取得する
    if len(sys.argv) != 1:
        global MEAN_RANGE, PATH_CSV_ACCELERATION_DATA, AMOUNT_OF_PLOT
        MEAN_RANGE = int(sys.argv[1])                                                   # 平均区間
        PATH_CSV_ACCELERATION_DATA = sys.argv[2]                                        # csvファイルのパス
        AMOUNT_OF_PLOT = sum([1 for _ in open(PATH_CSV_ACCELERATION_DATA)]) - 1         # プロット量
    # 1. csvファイル(加速度データ)を読み込み、pd.DataFrame型変数(df_read)を返す
    df_read = read_csv_(PATH_CSV_ACCELERATION_DATA).reset_index(drop='index')
    #df_read = df_read['onCrossroad']    # テスト: 列'onCrossroad'の抽出 (成功)
    #df_read = df_read[df_read['onCrossroad']=='0']    # 全ての交差点を抽出
    #df_read = df_read[df_read['crossroadID']==83]    # 交差点83を抽出
    df_read['time'] = pd.to_datetime(df_read['time'], format='%M:%S.%f')    # 列'time'をpd.datetime64[ns]型に変換
    time_for_assert_1 = df_read['time']                                     # アサーション用変数1: 関数plot_dataの呼び出し直前
    # 2. 上記で返されたdf_readについて、平均値を計算する(df_read)
    df_read = average_data(
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
    # 3. 隠れマルコフモデルを適用する
    if NUMBER_OF_ASSUMED_STATE > AMOUNT_OF_PLOT:  # 2021/7/5 2時頃: clustering, hmm共に、全く同じ例外が投げられることを確認した。
        raise Exception('確率モデルを用いる際に仮定する状態数の値が不適切です:(状態数, サンプル数)=({wrong_number_state}, {wrong_number_sample})'  \
                .format(wrong_number_state=NUMBER_OF_ASSUMED_STATE, wrong_number_sample=AMOUNT_OF_PLOT))
    elif NUMBER_OF_ASSUMED_STATE == AMOUNT_OF_PLOT    \
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
                .format(wrong_number_state=NUMBER_OF_ASSUMED_STATE, wrong_number_sample=AMOUNT_OF_PLOT))
    else:
        # 主成分分析をせずに、隠れマルコフモデルを適用する
        # [目的]: 次元削減でなく、データ可視化
        dict_param_original = estimate_state_data(
                input_df_read = df_read.drop('time', axis=1),
                input_how = ASSUMED_PROBABILISTIC_MODEL,
                input_number_of_assumed_state = NUMBER_OF_ASSUMED_STATE,
            )
    # 4. 主成分分析を実行する
    # FIXME2021/7/4: 上記の場合(main関数定義文下のif分岐)以外でも、切り出し区間によっては、関数decompose_dataで例外が発生する。
    # 例. (DATA_SAMPLED_FIRST, DATA_SAMPLED_LAST)=(5, 9)の時、ValueError: Shape of passed values is (4, 4), indices imply (4, 5)
    # [目的]: 次元削減でなく、データ可視化
    # - 2021.11.18の進捗報告時: 局所解に陥っている可能性があることを指摘された。
    df_read = df_read.loc[:, 'Acceleration(X)[g]':'AngularRate(Z)[dps]'].join(df_read['time']).reset_index(drop='index')
    df_pca, loading = decompose_data(df_read.drop('time', axis=1))
    df_pca = df_pca.join(df_read['time'])
    # 5. 上記の算出結果をプロットする
    time_for_assert_2 = df_read['time']                                             # アサーション用変数2: 列'time'をpd.datetime64[ns]型にキャストした直後
    assert time_for_assert_1.values.tolist() == time_for_assert_2.values.tolist()       # アサーション: 列'time'の値が、ここまでに誤って更新されていないか。
    plot_data(  # no-pca
            input_df_read = df_read,            # PCAしていないデータ
            input_dict_param = dict_param_original,     # [＊]: 次元削減でなくデータ可視化が目的のため、HMMは原データのみに適用
        )
    plot_data(  # pca
            input_df_read = df_pca,                 # PCAしたデータ
            input_dict_param = dict_param_original,
            input_loading = loading
        )

    # プロットの可視化
    # IPython環境でなくターミナル環境で実行する場合、プロットを可視化するのに必須
    # [関連]: decompose_data, plot_data
    #plt.show()

if __name__ == '__main__':
    main()
