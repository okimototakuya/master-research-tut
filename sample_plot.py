import csv
import matplotlib.pyplot as plt

x = []  # グラフの横軸(整数値)
y = []  # グラフの縦軸
EVAR = 0  # 整数値
#EVAR = 1  # 時刻
#TBAR = 2  # 鉛直方向の加速度(上が+)
#TBAR = 3  # 前後方向の加速度(前が+)
#TBAR = 4  # 左右方向の加速度(右が+)
TBAR = 5  # 左右回転の角速度(右が+)
#TBAR = 6  # 側方回転の角速度(右手側への回転が+)
#TBAR = 7  # 前傾後傾の角速度(後傾が+)

print("サンプルファイル名を入力してください↓")
str = input()
with open("./" + str) as f:

  reader = csv.reader(f)

  i = 0
  for row in reader:
    i = i + 1
    if i <= 3:
      pass
    else:
      if EVAR == 0:
        x.append(int(row[EVAR]))
      else:
        x.append(row[EVAR])
      y.append(float(row[TBAR]))

plt.plot(x,y)
plt.show()
