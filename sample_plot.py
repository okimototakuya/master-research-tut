import csv
import matplotlib.pyplot as plt

x = []  # グラフの横軸(整数値)
y = []  # グラフの縦軸

EVAR = 0  # 整数値
#EVAR = 1  # 時刻

#TVAR = 2  # 鉛直方向の加速度(上が+)
#TVAR = 3  # 前後方向の加速度(前が+)
#TVAR = 4  # 左右方向の加速度(右が+)
#TVAR = 5  # 左右回転の角速度(右が+)
#TVAR = 6  # 側方回転の角速度(右手側への回転が+)
TVAR = 7  # 前傾後傾の角速度(後傾が+)

print("サンプルファイル名を入力してください↓")
str = input()
with open("./" + str) as f:

  reader = csv.reader(f)

  i = 0
  for row in reader:
    i = i + 1
    if i <= 3:  # csvファイル:データに関与しない箇所は省略.
      pass
    else:
      if EVAR == 0: # 横軸が整数値
        x.append(int(row[EVAR]))
      else:         # 横軸が時刻表示
        x.append(row[EVAR])
      y.append(float(row[TVAR]))

if EVAR == 0:
  #strx = "整数値"
  strx = "Integer"
else:
  #strx = "時刻"
  strx = "Time"

if TVAR == 2:
  #stry = "鉛直方向の加速度(上が+)"
  stry = "Acceleration x[g]"
elif TVAR == 3:
  #stry = "前後方向の加速度(前が+)"
  stry = "Acceleration y[g]"
elif TVAR == 4:
  #stry = "左右方向の加速度(右が+)"
  stry = "Acceleration z[g]"
elif TVAR == 5:
  #stry = "左右回転の角速度(右が+)"
  stry = "AngularRate x[dps]"
elif TVAR == 6:
  #stry = "左右回転の角速度(右が+)"
  stry = "AngularRate y[dps]"
else:
  #stry ="前傾後傾の角速度(後傾が+)"
  stry = "AngularRate z[dps]"
 
plt.title(str)
plt.xlabel(strx)
plt.ylabel(stry)
plt.plot(x,y)
plt.show()
