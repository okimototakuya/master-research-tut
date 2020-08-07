import numpy as np
#import acceleration_plot as ap
import acceleration_plot2 as ap
from hmmlearn import hmm
import pandas as pd

# 予測値を格納する変数
pred = None
# 平均値をとる要素数
AVERAGE = 10

def hmmLearn():
	global pred
	# 加速度データのDataFrame型変数を作成.
	dataframe = ap.DataframeMaker(ap.filename)
	# 確率モデル(隠れマルコフモデルの作成.
	model = hmm.GaussianHMM(n_components=3, covariance_type="full")
	# DataFrame型変数から学習に用いる加速度データを抽出.
	X = (dataframe.df).loc[:,ap.acc[0]]
	X = pd.DataFrame(X)
	if len(ap.acc) > 1:
		for str in ap.acc[1:]:
			X_ = (dataframe.df).loc[:,str]
			X = X.join(X_)
	else:
		pass
	X = X.iloc[ap.HMM_RANGE_START:ap.HMM_RANGE_END, :]
	# 加速度の平均値を格納するためのDataFrame型変数
	X_ave = pd.DataFrame(np.arange(int(len(X)/AVERAGE)*len(X.columns)).reshape(int(len(X)/AVERAGE), len(X.columns)))
	for i in range(int(len(X)/AVERAGE)):
		print(X.iloc[i*AVERAGE:i*AVERAGE+AVERAGE, :].mean())
		X_ave.iloc[i, :] = pd.DataFrame(X.iloc[i*AVERAGE:i*AVERAGE+AVERAGE, :].mean())
	#model.fit(X)
	model.fit(X_ave)

	#np.set_printoptions(threshold=np.inf)		# 配列の要素を全て表示(状態系列)
	#print("初期確率\n", model.startprob_)
	#print("平均値\n", model.means_)
	#print("共分散値\n", model.covars_)
	#print("遷移確率\n", model.transmat_)
	#print("対数尤度\n", model.score(X))
	pred = model.predict(X)
	print("状態系列の復号\n", pred)

def main():
	hmmLearn()
	#getPred()

if __name__ == '__main__':
	main()
