# Name
主成分分析を用いて、隠れマルコフモデルによる状態推定値の根拠を視覚的に確認するプログラム


# Overview
本プログラムは、隠れマルコフモデルを用いて、児童の登下校時における運動データから行動要素を抽出することに応用される。


# Description
## /src/main
### 目的
---
プロダクトコード(ランタイム)を保管する。

### ファイル
---
#### acceleration_plot3.py
時系列運動データ(csv形式)を読み込み、主成分分析した後に隠れマルコフモデルによる状態推定を行うことで、  
状態推定値の根拠を視覚的に確認するためのプログラム

## /src/test
### 目的
---
テストスクリプトを保管する。

### ファイル
---
#### test_acceleration_plot3.py
プロダクトコードのテストを行う。テストの例として、データ読み込み時に意図した形式で読み込めたか確認するものや、  
データの一定区間での平均値算出が合っているか確認するものなどがある。
#### test_learn_api.py
Pythonのライブラリや特殊メソッドのテストを行う。

## /src/dataset
### 目的
---
時系列データを保管する。データはcsv形式で保管されている。txt形式について中身は同じである。


# Demo
同封の２つのpngファイルを参照のこと。


# Requirement
Anaconda4.10.1のbase環境上で、hmmlearn (PyPI)、seaborn (Anaconda Cloud)を新たに要する。


# Usage
プログラム実行時における各パラメータの値は、import文下の定数群の値によって調整すること。
## プロダクトコードの実行方法
src/mainに移動し、「python acceleration_plot3.py」
## テストスクリプトの実行方法
src/testに移動し、「python test_acceleration_plot3.py」
