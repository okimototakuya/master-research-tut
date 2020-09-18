import numpy as np
import pandas as pd
from hmmlearn import hmm
#import acceleration_plot as ap
import acceleration_plot2 as ap

# 予測値を格納する変数
#pred = None
# 平均値をとる要素数
AVERAGE = 10

def aveData(X):
    '加速度の平均値をとり、DataFrame型変数にして返す'
    # 加速度の平均値を格納するためのDataFrame型変数
    X_ave = pd.DataFrame(index=[], columns=ap.Global().acc[0])
    for i in range(int(len(X)/AVERAGE)):
        X_ave = X_ave.append(X.iloc[i*AVERAGE:i*AVERAGE+AVERAGE, :].mean(), ignore_index=True)
    return X_ave

def hmmLearn(df):
    'DataFrame型変数を引数にして、HMMによる学習を行う'
    #global pred
    # 加速度データのDataFrame型変数を作成.
    #dataframe = ap.DataframeMaker(ap.Global().filename[0])
    # 確率モデル(隠れマルコフモデルの作成.
    model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    # DataFrame型変数から学習に用いる加速度データを抽出.
    #X = (dataframe.df).loc[:, ap.Global().acc[0]]
    X = df.loc[:, (ap.Global().acc[0])[0]]
    #X = pd.DataFrame(X)
    if len(ap.Global().acc[0]) > 1:
        for ele in (ap.Global().acc[0])[1:]:
            X_ = df.loc[:, ele]
            X = X.join(X_)
    else:
        pass
    #X = X.iloc[ap.Global().HMM_RANGE_START:ap.Global().HMM_RANGE_END, :]
    # 加速度の平均値を格納するためのDataFrame型変数
    X_ave = aveData(X)
    #model.fit(X)
    model.fit(X_ave)

    #np.set_printoptions(threshold=np.inf)  # 配列の要素を全て表示(状態系列)
    #print("初期確率\n", model.startprob_)
    #print("平均値\n", model.means_)
    #print("共分散値\n", model.covars_)
    #print("遷移確率\n", model.transmat_)
    #print("対数尤度\n", model.score(X))
    #ap.Global().pred = model.predict(X)
    ap.Global().pred[0]= model.predict(X_ave)
    print("状態系列の復号\n", ap.Global().pred[0])

def main():
    'HMMによる学習を行う'
    hmmLearn()

if __name__ == '__main__':
    main()
