import pandas as pd
import matplotlib.pyplot as plt
from math import sin, radians

rad = 6.3781*10**6  # 地球半径[m]

"""
## ID16
# ファイル名
filename = "location_20181219_16.csv"
# 1日
#start = "2018-12-19 07:37:22+09"
#end = "2018-12-19 16:05:35+09"
# 登校
#start = "2018-12-19 07:37:22+09"
#end = "2018-12-19 07:47:10+09"
# 校内
#start = "2018-12-19 09:38:30+09"
#end = "2018-12-19 13:58:01+09"
# 下校
start = "2018-12-19 15:57:47+09"
end = "2018-12-19 16:05:35+09"
"""

## ID19
# ファイル名
filename = "location_20181219_19.csv"
# 1日
#start = "2018-12-19 07:41:13+09"
#end = "2018-12-19 16:07:49+09"
# 登校
#start = "2018-12-19 07:41:13+09"
#end = "2018-12-19 08:00:29+09"
# 校内
#start = "2018-12-19 09:38:05+09"
#end = "2018-12-19 13:58:05+09"
# 下校
start = "2018-12-19 15:57:47+09"
end = "2018-12-19 16:07:49+09"

def main():
  # 列名を明示的に指定することにより, 欠損値をNaNで補完.
  col_names = ['btx_id', 'timestamp', 
               'major', 'minor', 'pos_id',
               'deviceid', 'lat', 'lon',
               'recvDate',
              ]
  df = pd.read_csv(filename,
                   names=col_names,
                   ## 原データの時刻表示の余計な部分を削るため, 一旦str型で読込.
                   #parse_dates=['recvDate'], 
                   skiprows=1, 
                   index_col=8, # recvData(時刻)
                   #index_col=6,  # lat(緯度)
                   converters={'btx_id':int, 'timestamp':float, 
                               'major':int, 'minor':int, 'pos_id':int,
                               'deviceid':int, 'lat':float, 'lon':float,
                               'recvDate':str,
                              }
                  )

  # indexをpd.to_datetime()でDateTimeIndexにした.
  # df.loc[] : error → Jupyter実践入門(pandas0.19.2)と異なる
  # df.iloc[] : error
  df.index = [i[:19] for i in df.index] # recvDate:「+09」の削除
  df.index = pd.to_datetime(df.index) # DataFrame型のインデックス:string型 → DataTime型
  # Series型の各要素への関数適用:基本演算(+,-など) +,- / 一般的な関数(三角関数など) map
  df.loc[:, 'lat'] = (df.loc[:, 'lat']-35).map(radians).map(sin)
  df.loc[:, 'lon'] = (df.loc[:, 'lon']-137).map(radians).map(sin)

  df_time = df.loc[start:end]

  #ax1 = (df_time).plot(y=['lat']) 
  #ax2 = (df_time).plot(y='lon', secondary_y=['lat','lon'], ax=ax1) 
  #ax2.set_title(filename)
  #ax.set_ylabel('lat')
  #ax.right_ax.set_ylabel('lon')

  # DataFrame型plot() : xのラベル名(列名)は[]で囲まない. 
  # yはどっちでも.→ ラベル/リスト
  ax = (df_time).plot(x='lat', y=['lon'])

  plt.show()

if __name__ == '__main__':
  main()
