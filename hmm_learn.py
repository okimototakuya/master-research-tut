import numpy as np
import acceleration_plot as ap
from hmmlearn import hmm
import pandas as pd

global model
global X

class init():
	def getX(self):
		return self.X
	def getModel(self):
		return self.model

def main():
	acc = ap.dataframe_maker()
	acc.init()
	
	model = hmm.GaussianHMM(n_components=3, covariance_type="full")
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
	#print("状態系列の復号\n", model.predict(X))

if __name__ == '__main__':
	main()
