import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import acceleration_plot as ap 
import location_plot as lp

def main():
  dm_acc = ap.dataframe_maker()
  dm_acc.init()
  dm_loc = lp.dataframe_maker()
  dm_loc.init()
  dm_loc.makeLatLon()

  #print((dm_loc.df).loc[:,['lat','lon']])

  fig = plt.figure()
  #ax = fig.gca(projection='3d')
  ax = Axes3D(fig)
  #ax.plot((dm_loc.df).loc[:,['lat','lon']],3)

  x = (dm_loc.df).loc[:,['lat']]
  y = (dm_loc.df).loc[:,['lon']]
  z = (dm_acc.df).iloc[:len(dm_loc.df),1]  # 1:Acceleration_x
  ax.plot(x, y, z, color='green')
  #print(len(dm_loc.df))
  #print(len((dm_acc.df).iloc[:len(dm_loc.df),1]))

  #print((dm_acc.df).iloc[:5,1])
  #print(len(dm_loc.df))
  plt.show()

if __name__ == '__main__':
  main()
