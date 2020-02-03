import pandas as pd
import matplotlib
matplotlib.use('Agg')		# pyplotで生成した画像を保存するためのインポート
import matplotlib.pyplot as plt
import os
import hmm_learn
import numpy as np

## ID16
# ファイル名
filename = "dataset/LOG_20181219141837_00010533_0021002B401733434E45.csv"

## ID19
# ファイル名
#filename = "dataset/LOG_20181219141901_00007140_00140064401733434E45.csv"

## 画像ファイルの保存先
PATH = "/Users/okimototakuya/Library/Mobile Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/加速度・角加速度の時系列変化プロット"

class dataframe_maker():
	df = None # DataFrame型インスタンスを格納
	#global df

	def init(self):
		#global df
		# 列名を明示的に指定することにより, 欠損値をNaNで補完.
		col_names = ['line', 'time',
						'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
						'AngularRate_x', 'AngularRate_y', 'AngularRate_z',
						'Temperture', 'Pressure', 'MagnetCount', 'MagnetSwitch',
						]
		self.df = pd.read_csv(filename,
									names=col_names,
									skiprows=3,
									parse_dates=['time'],
									index_col=0, # 0:整数値, 1:時刻
									converters={'line':int, 'time':str,
													'Acceleration_x':float, 'Acceleration_y':float, 'Acceleration_z':float,
													'AngularRate_x':float, 'AngularRate_y':float, 'AngularRate_z':float,
													'Temperture':float, 'Pressure':float, 'MagnetCount':int, 'MagnetSwitch':int,
													}
									)

class dataframe_plotter():
	"""
	def plotTimeAcc(self, df):
		## 加速度の時系列変化をプロット
		ax1 = df.plot(y='Acceleration_x')
		ax2 = df.plot(y='Acceleration_y', ax=ax1)
		ax3 = df.plot(y='Acceleration_z', ax=ax2)
		ax3.set_title(filename)
		plt.show()

	def plotTimeAng(self, df):
		## 角速度の時系列変化をプロット
		ax1 = df.plot(y='AngularRate_x')
		ax2 = df.plot(y='AngularRate_y', ax=ax1)
		ax3 = df.plot(y='AngularRate_z', ax=ax2)
		ax3.set_title(filename)
		plt.show()
	"""

	#def plotTimeAccAng(self, df, delta, acc_x=None, acc_y=None, acc_z=None, ang_x=None, ang_y=None, ang_z=None):
	def plotTimeAccAng(self, df, delta, *args):
		global pred
		predict = pd.DataFrame(pred, columns=['pred'])
		df = pd.concat([df[list(args)], predict], axis=1)
		copy_df = df
		#print(copy_df)
		## 加速度・角速度の時系列変化をプロット
		for i in range(500):
			for arg in args:
				copy_df.loc[:, arg] = df.loc[delta*i:delta*(i+1), arg]		# 加速度x

			copy_df.loc[:, 'pred'] = df.loc[delta*i:delta*(i+1), 'pred']			# 予測値x

			ax1 = copy_df.plot(y='Acceleration_x')
			#ax2 = copy_df.plot(y='Acceleration_y', ax=ax1)
			#ax3 = copy_df.plot(y='Acceleration_z', ax=ax2)
			ax2 = copy_df.plot(y='AngularRate_x', ax=ax1)
			#ax5 = copy_df.plot(y='AngularRate_y', ax=ax4)
			#ax6 = copy_df.plot(y='AngularRate_z', secondary_y=['Acceleration_x','AngularRate_x'], ax=ax5)

			ax_pred = copy_df.plot(y='pred', ax=ax2)
			ax_pred.set_title(filename)
			#plt.show()
			plt.savefig(os.path.join(PATH, "demo"+str(i)+".png"))


def main():
	global pred
	#np.set_printoptions(threshold=np.inf)		# 配列の要素を全て表示(状態系列)
	pred = hmm_learn.getPred()

	dm = dataframe_maker()
	dm.init()
	dp = dataframe_plotter()
	#dp.plotTimeAcc(dm.df)
	#dp.plotTimeAng(dm.df)
	dp.plotTimeAccAng(dm.df, 250, 'Acceleration_x', 'AngularRate_x')
	#dp.plotTimeAccAng(dm.df, 250, 1, 4)

if __name__ == '__main__':
	pred = None		# 予測値を取得する変数.
	#df = None		# DataFrame型を格納する変数.
	main()
