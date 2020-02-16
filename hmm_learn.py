import numpy as np
import acceleration_plot as ap
from hmmlearn import hmm
import pandas as pd

def getPred():
	global pred
	hmmLearn()
	return pred

def hmmLearn():
	global pred
	global filename
	# 加速度データのDataFrame型変数を作成.
	acc = ap.dataframe_maker(filename)
	#acc.init()
	# 確率モデル(隠れマルコフモデルの作成.
	model = hmm.GaussianHMM(n_components=3, covariance_type="full")
	# DataFrame型変数から学習に用いる加速度データを抽出.
	X1 = (acc.df).iloc[:,1]
	X1 = pd.DataFrame(X1)
	X2 = (acc.df).iloc[:,4]
	X = X1.join(X2)
	model.fit(X)

	#np.set_printoptions(threshold=np.inf)		# 配列の要素を全て表示(状態系列)
	#print("初期確率\n", model.startprob_)
	#print("平均値\n", model.means_)
	#print("共分散値\n", model.covars_)
	#print("遷移確率\n", model.transmat_)
	#print("対数尤度\n", model.score(X))
	pred = model.predict(X)
	print("状態系列の復号\n", pred)

def main():
	#hmmLearn()
	getPred()

if __name__ == '__main__':
# 予測値を格納する変数
	pred = None
## ID16
# ファイル名
	filename = "dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"
## ID19
# ファイル名
	#filename = "dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"
# main関数の実行
	main()
