# Name
主成分分析を用いて、隠れマルコフモデルによる状態推定値の根拠を視覚的に確認するプログラム


# Overview
本プログラムは、隠れマルコフモデルを用いて、時系列データから状態を抽出することに応用される。


# Description
## /src/main
### 目的
---
プロダクトコード(ランタイム)を保管する。

### ファイル
---
#### acceleration_plot3.py
時系列データ(csv形式)を読み込み、主成分分析した後に隠れマルコフモデルによる状態推定を行うことで、  
状態推定値の根拠を視覚的に確認するためのプログラム

## /src/test
### 目的
---
テストスクリプトを保管する。

### ファイル
---
#### test_"hoge-hoge".py
プロダクトコードのテストを行う。テストの例として、データ読み込み時に意図した形式で読み込めたか確認するものや、  
データの一定区間での平均値算出が合っているか確認するものなどがある。
#### test_learn_api.py
Pythonのライブラリや特殊メソッドのテストを行う。

## /src/dataset
### 目的
---
- 時系列データを保管する。データはcsv形式で保管されている。txt形式について中身は同じである。
- ID16: 全交差点番号リスト:[2, 96, 32, 33, 34, 73, 74, 79, 82, 83, 84, 46, 45] (src/test/pickup_cross_label.pyより)
-### サンプルデータの読み方
----
-#### LOG_2018\*.csv (加速度・角速度情報)
----
-##### 加速度(Acceleration)
-- x鉛直(上が+)
-- y前後(前が+)
-- z左右(右が+)
-##### 角速度(AngularRate)
-- xヨー   左右回転(右が+)
-- yロール 側方回転(右手側への回転が+)
-- zピッチ 前傾後傾(後傾が+)

# Demo
同封の２つのpngファイルを参照のこと。


# Requirement
- Anaconda4.10.1のbase環境上で、hmmlearn (PyPI)、seaborn (Anaconda Cloud)を新たに要する。
- プロジェクトの最上位ディレクトリに、dataset/\*.csvを配置すること。


# Usage
プログラム実行時における各パラメータの値は、import文下の定数群の値によって調整すること。
## プロダクトコードの実行方法
- コマンドライン引数を与えない場合
-- src/mainに移動し、「python acceleration_plot3.py」
- コマンドライン引数を与える場合
-- src/mainに移動し、「python acceleration_plot3.py [平均区間(整数)] [csvファイルの相対パス(クオーテーションは不要)]」

## テストスクリプトの実行方法
src/testに移動し、「python test_"hoge-hoge".py」
