import pandas as pd
import matplotlib
#matplotlib.use('Agg')		# pyplotで生成した画像を保存するためのインポート
import matplotlib.pyplot as plt
import os
import hmm_learn
import cluster_learn
import numpy as np
import sys

## この位置でグローバル変数扱いになる.
## 予測値を格納する変数
pred = None
## ID16
# ファイル名
filename = "dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"
## ID19
# ファイル名
#filename = "dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"
## 加速度のリスト
acc = [
		'Acceleration_x',
		'Acceleration_y',
		'Acceleration_z',
#		'AngularRate_x',
#		'AngularRate_y',
#		'AngularRate_z',
		]
## 画像ファイルの保存先
PATH = "/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/ID16/hmm1x1y1z70000-80000_100"
#PATH = "/Users/okimototakuya/Desktop/tmp"
## 一つのグラフのプロット数
#PLOT_SEG = 10000
PLOT_SEG = 131663
## 隠れマルコフモデルを適用させる範囲
#HMM_RANGE_START = 70000
#HMM_RANGE_END = 80000
HMM_RANGE_START = 0
HMM_RANGE_END = 131663

class DataframeMaker():
	def __init__(self, filename):
		# 列名を明示的に指定することにより, 欠損値をNaNで補完.
		col_names = ['line', 'time',
						'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
						'AngularRate_x', 'AngularRate_y', 'AngularRate_z', 'Temperture', 'Pressure', 'MagnetCount', 'MagnetSwitch',
						]
		self.df = pd.read_csv(filename,
									names=col_names,
									skiprows=3,
									parse_dates=['time'],
									index_col='time',
									converters={'line':int, 'time':str,
													'Acceleration_x':float, 'Acceleration_y':float, 'Acceleration_z':float,
													'AngularRate_x':float, 'AngularRate_y':float, 'AngularRate_z':float,
													'Temperture':float, 'Pressure':float, 'MagnetCount':int, 'MagnetSwitch':int,
													}
									)

class DataframePlotter():
	@staticmethod
	def plot(df, delta, args):		# delta:グラフの定義域,*args:グラフを描く列のタプル(＊タプルで受け取る)
		if sys.argv[1] != '2':
			global pred
			df = df.iloc[HMM_RANGE_START:HMM_RANGE_END, :].reset_index()
			predict = pd.DataFrame(pred, columns=['pred'])
			df = pd.concat([df[list(args)], predict], axis=1)
			## 加速度・角速度の時系列変化をプロット
			for i in range(int(len(df)/delta)):
				copy_df = df.loc[delta*i:delta*(i+1), :]
				copy_df.dropna(how='all')
				ax1 = copy_df[list(args)].plot()
				#if i == 3:
				#	break
				#ax = copy_df[['pred']].plot.bar(ax=ax1, width=1.0)
				ax = copy_df[['pred']].plot(ax=ax1)
				ax.set_title(filename)
				ax.set_ylim([-5.0, 2.5])
				#plt.show()
				plt.savefig(os.path.join(PATH, "demo"+str(i)+".png"))
		else:
			df = df.iloc[HMM_RANGE_START:HMM_RANGE_END, :].reset_index()		# 2次元プロットする範囲を指定

			df = hmm_learn.aveData(df)			# 加速度データを平均化
			delta = int(delta/hmm_learn.AVERAGE)		# 平均値をとる要素数で区間を割る

			for i in range(int(len(df)/delta)):
				copy_df = df.iloc[delta*i:delta*(i+1), :]
				copy_df.dropna(how='all')
				#ax = copy_df.plot(x=acc[0], y=acc[1])			# 折れ線グラフ
				ax = copy_df.plot.scatter(x=acc[0], y=acc[1])		# 散布図
				ax.set_title(filename)
				ax.set_xlim([-5.5, 1.0])
				ax.set_ylim([-2.5, 2.0])
				plt.show()

def main():
	global filename
	global PATH
	global pred
	global acc
	global PLOT_SEG

	if sys.argv[1] == '0':		# 隠れマルコフモデル
		#np.set_printoptions(threshold=np.inf)		# 配列の要素を全て表示(状態系列)
		hmm_learn.hmmLearn()
		pred = hmm_learn.pred
	elif sys.argv[1] == '1':		# クラスタリング
		#np.set_printoptions(threshold=np.inf)		# 配列の要素を全て表示(状態系列)
		cluster_learn.clusterLearn()
		pred = cluster_learn.pred
	elif sys.argv[1] == '2':		# 加速度を２次元プロット
		pass
	else:
		print("Error is here.")

	dataframe = DataframeMaker(filename)
	DataframePlotter.plot(dataframe.df, PLOT_SEG, tuple(acc))

if __name__ == '__main__':
	main()