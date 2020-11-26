import numpy as np
import pandas as pd
from hmmlearn import hmm
import config
#import acceleration_plot as ap
import acceleration_plot2 as ap


def hmmLearn(df):
    'DataFrame型変数を引数にして、HMMによる学習を行う'
    #global pred
    # 加速度データのDataFrame型変数を作成.
    #data_sampled_by_func = ap.DataframeMaker(config.data_read_by_api)
    # 確率モデル(隠れマルコフモデルの作成.
    model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    # DataFrame型変数から学習に用いる加速度データを抽出.
    #X = (data_sampled_by_func.df).loc[:, config.features_selected_manually]
    print(type(df))
    X = df.loc[:, (config.features_selected_manually)[0]]
    X = pd.DataFrame(X)
    if len(config.features_selected_manually) > 1:
        for ele in (config.features_selected_manually)[1:]:
            X_ = df.loc[:, ele]
            X = X.join(X_)
    else:
        pass
    #X = X.iloc[config.HMM_RANGE_START:config.HMM_RANGE_END, :]
    # 加速度の平均値を格納するためのDataFrame型変数
    X_ave = config.aveData(X)
    #model.fit(X)
    model.fit(X_ave)

    #np.set_printoptions(threshold=np.inf)  # 配列の要素を全て表示(状態系列)
    #print("初期確率\n", model.startprob_)
    #print("平均値\n", model.means_)
    #print("共分散値\n", model.covars_)
    #print("遷移確率\n", model.transmat_)
    #print("対数尤度\n", model.score(X))
    #config.pred_by_prob_model = model.predict(X)
    config.pred_by_prob_model = model.predict(X_ave)
    config.data_sampled_by_func = X_ave
    print("状態系列の復号\n", config.pred_by_prob_model)


##################################################################
##########メイン関数##############################################
##################################################################
def main():
    'HMMによる学習を行う'
    hmmLearn()

if __name__ == '__main__':
    main()
