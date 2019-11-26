import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

fig = plt.figure(figsize=(4.5, 2))
ax = fig.add_subplot(111)

n = 20
x = np.arange(n)
y = np.ones(n)
#z = x*y
z = np.random.rand(n)
x=pd.DataFrame(x)
y=pd.DataFrame(y)
z=pd.DataFrame(z)
"""
for i in range(n):
  color = cm.Set1(i / n*0.5)
  ax.scatter(x[i], y[i], c=color)
"""
color = "inferno"
ax.scatter(x, y, c=z)
#ax.scatter(x, y, cmap=color)

print(color)
plt.show()
