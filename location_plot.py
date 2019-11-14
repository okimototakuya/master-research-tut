import pandas as pd
import matplotlib.pyplot as plt

def main():
  print("サンプルファイル名を入力してください")
  print(":", end="")
  filename = input()

  col_names = ['btx_id', 'timestamp', 
               'major', 'minor', 'pos_id',
               'deviceid', 'lat', 'lon',
               'recvDate',
              ]
  df = pd.read_csv(filename,
                   names=col_names,
                   #parse_dates=['recvDate'],
                   skiprows=1, 
                   index_col=8,
                   converters={'btx_id':int, 'timestamp':float, 
                               'major':int, 'minor':int, 'pos_id':int,
                               'deviceid':int, 'lat':float, 'lon':float,
                               'recvDate':str,
                              }
                  )

  df.index = [i[:19] for i in df.index] # recvDate:「+09」の削除
  df.index = pd.to_datetime(df.index) # DataFrame型のインデックス:string型 → DataTime型
  ax1 = df.plot(y='lat') 
  ax2 = df.plot(y='lon', secondary_y=['lat','lon'], ax=ax1) 
  ax2.set_title(filename)
  #ax.set_ylabel('lat')
  #ax.right_ax.set_ylabel('lon')
  plt.show()

if __name__ == '__main__':
  main()
