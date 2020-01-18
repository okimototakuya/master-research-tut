import numpy as np
import acceleration_plot as ap
from hmmlearn import hmm
import pandas as pd

acc = ap.dataframe_maker()
acc.init()

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
X1 = (acc.df).iloc[:,1]
X1 = pd.DataFrame(X1)
X2 = (acc.df).iloc[:,4]
X = X1.join(X2)
model.fit(X)
