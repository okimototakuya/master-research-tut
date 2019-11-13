import pandas as pd
import matplotlib.pyplot as plt

def main():
  print("サンプルファイル名を入力してください↓")
  filename = input()

  col_names = ['line', 'time', 
               'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
               'AngularRate_x', 'AngularRate_y', 'AngularRate_z',
               'Temperture', 'Pressure', 'MagnetCount', 'MagnetSwitch',
              ]
  df = pd.read_csv(filename,
                   names=col_names,
                   skiprows=3, 
                   #index_col=0,
                   converters={'line':int, 'time':str, 
                               'Acceleration_x':float, 'Acceleration_y':float, 'Acceleration_z':float,
                               'AngularRate_x':float, 'AngularRate_y':float, 'AngularRate_z':float,
                               'Temperture':float, 'Pressure':float, 'MagnetCount':int, 'MagnetSwitch':int,}
                  )
  #print(df)
  ax = df.plot(x='time', y='Acceleration_x') 
  #ax = df.plot(x='time', y='AngularRate_x') 
  ax.set_title(filename)
  plt.show()

if __name__ == '__main__':
  main()
