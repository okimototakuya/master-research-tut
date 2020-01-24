import pandas as pd
import matplotlib
matplotlib.use('Agg')		# pyplotで生成した画像を保存するためのインポート
import matplotlib.pyplot as plt
import os

## ID16
# ファイル名
filename = "LOG_20181219141837_00010533_0021002B401733434E45.csv"

## ID19
# ファイル名
#filename = "LOG_20181219141901_00007140_00140064401733434E45.csv"

## 画像ファイルの保存先
#PATH = "/Users/okimototakuya/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/研究/M1/研究データ/サンプル2件/"
PATH = "/Users/okimototakuya/Desktop"

class dataframe_maker():
	df = None # DataFrame型インスタンスを格納

	def init(self):
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

	def plotTimeAccAng(self, df):
		## 加速度・角速度の時系列変化をプロット
		ax1 = df.plot(y='Acceleration_x')
		ax2 = df.plot(y='Acceleration_y', ax=ax1)
		ax3 = df.plot(y='Acceleration_z', ax=ax2)
		ax4 = df.plot(y='AngularRate_x', ax=ax3)
		ax5 = df.plot(y='AngularRate_y', ax=ax4)
		ax6 = df.plot(y='AngularRate_z', secondary_y=['Acceleration_x','AngularRate_x'], ax=ax5)
		ax6.set_title(filename)
		#plt.show()
		plt.savefig(os.path.join(PATH, "demo.png"))


def main():
	dm = dataframe_maker()
	dm.init()
	dp = dataframe_plotter()
	#dp.plotTimeAcc(dm.df)
	#dp.plotTimeAng(dm.df)
	dp.plotTimeAccAng(dm.df)

if __name__ == '__main__':
	main()
