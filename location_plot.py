import pandas as pd
import matplotlib.pyplot as plt

## ID16
# ファイル名
filename = "location_20181219_16.csv"
# 1日
start = "2018-12-19 07:37:22+09"
end = "2018-12-19 16:05:35+09"
# 登校
#start = "2018-12-19 07:37:22+09"
#end = "2018-12-19 07:47:10+09"
# 下校
#start = "2018-12-19 15:57:47+09"
#end = "2018-12-19 16:05:35+09"

def main():
  #print("サンプルファイル名を入力してください")
  #print(":", end="")
  #filename = input()

  #print("表示する時間帯を入力してください")
  #print(":", end="")
  #time = input()
  #print(filename)
  #print(time)

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
  #print(df.index[:6])
  #print(df.loc[07:37:22:08:00:00 ,'lat'])
  #print(df.loc['2018-12-19 07:37:22':'2018-12-19 07:47:10', 'lat'])
  #print(df.index['2018-12-19 07:37:22':'2018-12-19 07:47:10'])
  #print(df.index[0:10])
  #print(df.loc[:'2018-12-19 07:47:10'])
  df_time = df.loc[start:end]
  #print(df_time)

  ax1 = (df_time).plot(y=['lat']) 
  ax2 = (df_time).plot(y='lon', secondary_y=['lat','lon'], ax=ax1) 
  ax2.set_title(filename)
  #ax.set_ylabel('lat')
  #ax.right_ax.set_ylabel('lon')
  plt.show()

if __name__ == '__main__':
  main()
