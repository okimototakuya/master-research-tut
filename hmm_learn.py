import numpy as np
import acceleration_plot as ap
from hmmlearn import hmm
import pandas as pd

def getPred():
	global pred
	hmmLearn()
	return pred

def hmmLearn():
	global model
	global X
	global pred
	model.fit(X)
	np.set_printoptions(threshold=np.inf)		# 配列の要素を全て表示(状態系列)
	#print("初期確率\n", model.startprob_)
	#print("平均値\n", model.means_)
	#print("共分散値\n", model.covars_)
	#print("遷移確率\n", model.transmat_)
	#print("対数尤度\n", model.score(X))
	pred = model.predict(X)
	print("状態系列の復号\n", pred)

def main():
	hmmLearn()

if __name__ == '__main__':
	# 加速度データのDataFrame型変数を作成.
	acc = ap.dataframe_maker()
	acc.init()
	# 確率モデル(隠れマルコフモデルの作成.
	model = hmm.GaussianHMM(n_components=3, covariance_type="full")
	# DataFrame型変数から学習に用いる加速度データを抽出.
	X1 = (acc.df).iloc[:,1]
	X1 = pd.DataFrame(X1)
	X2 = (acc.df).iloc[:,4]
	X = X1.join(X2)
	# 予測値を格納する変数
	pred = None
	# main関数の実行
	main()
