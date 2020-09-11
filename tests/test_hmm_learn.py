import unittest
import hmm_learn
import acceleration_plot2 as ap
import pandas as pd

class TestHmm_learn(unittest.TestCase):

	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_aveData(self):
		# 加速度データのDataFrame型変数を作成.
		dataframe = ap.DataframeMaker(ap.filename)
		# DataFrame型変数から学習に用いる加速度データを抽出.↲
		X = (dataframe.df).loc[:,ap.acc[0]]
		X = pd.DataFrame(X)
		if len(ap.acc) > 1:
			for str in ap.acc[1:]:
				X_ = (dataframe.df).loc[:,str]
				X = X.join(X_)
		else:
			pass
		X = X.iloc[ap.HMM_RANGE_START:ap.HMM_RANGE_END, :]
		length = len(hmm_learn.aveData(X))
		self.assertEqual(length, 1000)

if __name__ == '__main__':
	unittest.main()
