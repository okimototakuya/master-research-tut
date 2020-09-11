import csv
import matplotlib.pyplot as plt

x = []  # グラフの横軸(整数値)
y1 = []  # グラフの縦軸x
y2 = []  # グラフの縦軸y
y3 = []  # グラフの縦軸z

EVAR = 0  # 整数値
#EVAR = 1  # 時刻

TVAR = 0  # 加速度[g]
#TVAR = 1  # 角速度[dps]

print("サンプルファイル名を入力してください↓")
str = input()
with open("./" + str, mode='r') as f:
	reader = list(csv.reader(f))  # csv.readerはインデックス作成をサポートしていない.
	for row in reader[3:]:
		if EVAR == 0: # 横軸が整数値
			x.append(int(row[EVAR]))
		else:         # 横軸が時刻表示
			x.append(row[EVAR])

		if TVAR == 0: # 縦軸が加速度[g]
			y1.append(float(row[2]))
			y2.append(float(row[3]))
			y3.append(float(row[4]))
		else:         # 縦軸が角速度[dps]
			y1.append(float(row[5]))
			y2.append(float(row[6]))
			y3.append(float(row[7]))

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

plt.title(str)
plt.xlabel(strx)
plt.ylabel(stry)
plt.plot(x,y1,label="x")
plt.plot(x,y2,label="y")
plt.plot(x,y3,label="z")
plt.legend()  # 凡例の表示
plt.show()
