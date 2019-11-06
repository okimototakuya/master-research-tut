import csv
import matplotlib.pyplot as plt

x = []  # グラフの横軸(整数値)
y = []  # グラフの縦軸

with open("./LOG_20181219141837_00010533_0021002B401733434E45.csv") as f:
  #print(f.read())

  reader = csv.reader(f)
  #reader = reader[3:]
  #print(reader)

  #del reader[:3]

  i = 0
  for row in reader:
    i = i + 1
    #print(row[0])
    if i <= 3:
      pass
    else:
      x.append(int(row[0]))
      y.append(float(row[2]))
  #del x[:3]
  #del y[:3]
  #del x[10:]
  #del y[10:]
  #print(y)
  #x = int(x)
  #y = int(y)

plt.plot(x,y)
plt.show()
#print(len(x))
#print(len(y))
