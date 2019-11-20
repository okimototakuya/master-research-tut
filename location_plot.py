import pandas as pd
import matplotlib.pyplot as plt
from math import sin, radians

rad = 6.3781*10**6  # 地球半径[m]

## ID16
# ファイル名
filename = "location_20181219_16.csv"
# 1日
start = "2018-12-19 07:37:22+09"
end = "2018-12-19 16:05:35+09"
# 登校
#start = "2018-12-19 07:37:22+09"
#end = "2018-12-19 07:47:10+09"
# 校内
#start = "2018-12-19 09:38:30+09"
#end = "2018-12-19 13:58:01+09"
# 下校
#start = "2018-12-19 15:57:47+09"
#end = "2018-12-19 16:05:35+09"

"""
## ID19
# ファイル名
filename = "location_20181219_19.csv"
# 1日
start = "2018-12-19 07:41:13+09"
end = "2018-12-19 16:07:49+09"
# 登校
#start = "2018-12-19 07:41:13+09"
#end = "2018-12-19 08:00:29+09"
# 校内
#start = "2018-12-19 09:38:05+09"
#end = "2018-12-19 13:58:05+09"
# 下校
#start = "2018-12-19 15:57:47+09"
#end = "2018-12-19 16:07:49+09"
"""

class dataframe_maker():
  df = None # DataFrame型インスタンスを格納

  def init(self):
    # 列名を明示的に指定することにより, 欠損値をNaNで補完.
    col_names = ['btx_id', 'timestamp', 
                 'major', 'minor', 'pos_id',
                 'deviceid', 'lat', 'lon',
                 'recvDate',
                ]
    self.df = pd.read_csv(filename,
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
    (self.df).index = [i[:19] for i in (self.df).index] # recvDate:「+09」の削除
    (self.df).index = pd.to_datetime((self.df).index) # DataFrame型のインデックス:string型 → DataTime型

  def makeTimeLatLon(self):
    self.df = (self.df).loc[start:end]
        
  def makeLatLon(self):
    # Series型の各要素への関数適用:基本演算(+,-など) +,- / 一般的な関数(三角関数など) map
    (self.df).loc[:, 'lat'] = ((self.df).loc[:, 'lat']-35).map(radians).map(sin)
    (self.df).loc[:, 'lon'] = ((self.df).loc[:, 'lon']-137).map(radians).map(sin)
    self.df = (self.df).loc[start:end]


class dataframe_plotter():
  def plotTimeLatLon(self, df):
    ## 緯度(lat)経度(lon)の時系列変化をプロット
    ax1 = df.plot(y=['lat']) 
    ax2 = df.plot(y='lon', secondary_y=['lat','lon'], ax=ax1) 
    ax2.set_title(filename)
    #ax.set_ylabel('lat')
    #ax.right_ax.set_ylabel('lon')
    plt.show()

  def plotLatLon(self, df):
    ## 緯度(lat)経度(lon)の射影から,２次元位置座標をプロット
    # DataFrame型plot() : xのラベル名(列名)は[]で囲まない. 
    # yはどっちでも.→ ラベル/リスト
    ax = df.plot(x='lat', y=['lon'])
    plt.show()


def main():
  #plotTimeLatLon(df) # 緯度(lat)経度(lon)の時系列変化をプロット
  #plotLatLon(df)  # 緯度(lat)経度(lon)の射影から,２次元位置座標をプロット
  dm = dataframe_maker()
  dm.init()
  dm.makeLatLon()
  dp = dataframe_plotter()
  dp.plotLatLon(dm.df)

if __name__ == '__main__':
  main()
