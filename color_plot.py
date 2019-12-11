import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import acceleration_plot as ap
import location_plot as lp

def main():
	dm_acc = ap.dataframe_maker()
	dm_acc.init()
	dm_loc = lp.dataframe_maker()
	dm_loc.init()
	dm_loc.makeLatLon()

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)

	x = (dm_loc.df).loc[:,['lat']]
	y = (dm_loc.df).loc[:,['lon']]
	## Acceleration_xのみ
	z = (dm_acc.df).iloc[:len(x),1]  # 1:Acceleration_x
	z = round(-z,4)		#加速度を正にし, 有効数字2桁で丸める(rgb値(0~128)に対応)
	z.loc[z>1] = 1			#加速度が１を超えたものは１を代入し,
	z.loc[z<0] = 0			#０を下回ったものは０を代入する.

	for i in range(len(x)):
		#グレースケール:0.0黒~1.0白
		ax.scatter(x.iloc[i], y.iloc[i], label='Gray scale', color=str(z.iloc[i]))

	#print(color)
	## zに「Acceleration_x」を指定するとエラーが出る.
	#ax.scatter(x,y,c=z)
	#ax.scatter(x,y)
	
	#fig.colorbar(im)
	#plt.colorbar()

	plt.xlim(-0.0001, 0.0001)
	plt.ylim(-0.0001, 0.0001)
	plt.show()

if __name__ == '__main__':
	main()
