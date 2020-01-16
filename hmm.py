import numpy as np
import acceleration_plot as ap
from hmmlearn import hmm

acc = ap.dataframe_maker()
acc.init()
#print((acc.df).iloc[:,1])		# 1:Acceleration_x

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
X = (acc.df).iloc[:,1]
model.fit(X)
