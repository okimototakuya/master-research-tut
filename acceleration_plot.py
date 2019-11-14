import pandas as pd
import matplotlib.pyplot as plt

def main():
  print("サンプルファイル名を入力してください")
  print(":", end="")
  filename = input()

  col_names = ['line', 'time', 
               'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
               'AngularRate_x', 'AngularRate_y', 'AngularRate_z',
               'Temperture', 'Pressure', 'MagnetCount', 'MagnetSwitch',
              ]
  df = pd.read_csv(filename,
                   names=col_names,
                   skiprows=3, 
                   index_col=1, # 0:整数値, 1:時刻
                   converters={'line':int, 'time':str, 
                               'Acceleration_x':float, 'Acceleration_y':float, 'Acceleration_z':float,
                               'AngularRate_x':float, 'AngularRate_y':float, 'AngularRate_z':float,
                               'Temperture':float, 'Pressure':float, 'MagnetCount':int, 'MagnetSwitch':int,
                              }
                  )
  df.index = pd.to_datetime(df.index, format='%H:%M:%S.%f')
  #ax1 = df.plot(y='Acceleration_x') 
  #ax2 = df.plot(y='Acceleration_y', ax=ax1) 
  #ax3 = df.plot(y='Acceleration_z', ax=ax2) 
  #ax1 = df.plot(y='AngularRate_x') 
  #ax2 = df.plot(y='AngularRate_y', ax=ax1) 
  #ax3 = df.plot(y='AngularRate_z', ax=ax2) 
  ax1 = df.plot(y='Acceleration_x') 
  ax2 = df.plot(y='Acceleration_y', ax=ax1) 
  ax3 = df.plot(y='Acceleration_z', ax=ax2) 
  ax4 = df.plot(y='AngularRate_x', ax=ax3) 
  ax5 = df.plot(y='AngularRate_y', ax=ax4) 
  ax6 = df.plot(y='AngularRate_z', secondary_y=['Acceleration_x','AngularRate_x'], ax=ax5) 
  ax6.set_title(filename)
  plt.show()

if __name__ == '__main__':
  main()
