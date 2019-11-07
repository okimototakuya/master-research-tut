import csv
import matplotlib.pyplot as plt

x = []  # グラフの横軸(整数値)
y = []  # グラフの縦軸

EVAR = 0  # 整数値
#EVAR = 1  # 時刻

TVAR = 0  # 加速度
#TVAR = 1  # 角速度

print("サンプルファイル名を入力してください↓")
str = input()
with open("./" + str, mode='r') as f:

  reader = list(csv.reader(f))  # csv.readerはインデックス作成をサポートしていない.

  for row in reader[3:]:
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

if TVAR == 0:
  #stry = "加速度x/y/z[g]"
  stry = "Acceleration x/y/z[g]"
else:
  #stry = "角速度x/y/z[dps]"
  stry = "AngularRate x/y/z[dps]"
 
plt.title("サンプル名:"+str)
plt.xlabel(strx)
plt.ylabel(stry)
plt.plot(x,y)
plt.show()
