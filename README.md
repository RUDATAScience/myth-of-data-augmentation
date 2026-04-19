# Oversampling Paradox Simulator: 歪んだデータにおけるデータ拡張の自己矛盾

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本リポジトリは、アンケートや大規模データ収集において「社会的望ましさバイアス（忖度）」が発生している際、不均衡データに対する標準的な救済策である**オーバーサンプリング（SMOTEや単純複製など）がいかに無意味であり、かつ有害であるか**を証明するPythonシミュレーションです。

機械学習の現場における「データが足りないなら水増しすればよい」という常識が、情報生成プロセスが歪んでいる系においては「純粋なノイズの神格化」へと変質するプロセスを定量化します。

## 📌 背景と問題意識 (Background)

データ分析において、極端なマイノリティの声（例えば「評価1」）が少ない場合、実務家はデータを補完・増幅することでAIモデルの学習精度を向上させようと試みます。

しかし、本シミュレーションは以下の残酷な事実（オーバーサンプリングのパラドックス）を証明します。

1. **真実の逃亡 (Fugitives)**: 
   同調圧力が強い環境では、本来「評価1」をつけるべき真のマイノリティの大多数は、すでに忖度して「評価3」や「評価4」の中に隠れて（逃亡して）しまっています。
2. **シグナル・ノイズ比の逆転 (SNR < 1)**: 
   忖度後にデータ上に残っているわずかな「評価1」は、真実を貫いた少数の人々と、マジョリティがランダムに間違えた「純粋なノイズ（回答ミス等）」が混在しており、多くの場合ノイズの方が数が多くなります。
3. **情報の偽造 (Algorithmic Pollution)**: 
   この状態で評価1をいくら水増ししても、それは「消えた不満層」を復元することにはなりません。単にノイズと極端な意見を増幅させ、現実には存在しない「偽のマイノリティ像」をAIに学習させる最悪のアルゴリズム汚染を引き起こします。

## 🧮 数理モデル (Mathematical Model)

個人の最終的な効用（U_total）を、内発的な「本音（U_true）」と外発的な「忖度（U_target）」の線形結合として定義し、Softmax関数を通じて選択確率を算出します。

U_total = (1 - v2) * U_true + v2 * U_target

* v2: 社会的望ましさ（忖度）の重み。本実験では臨界点の 0.5 に設定。
* Beta: 回答者の確信度。
* シミュレーション規模: 全体10000人のうち、真のマイノリティを10%（1000人）として設定し、忖度後の回答行動をモンテカルロ法で再現。

## 📊 出力される分析結果 (Outputs)

スクリプトを実行すると `oversampling_paradox_results` ディレクトリが作成され、以下の高解像度グラフ（PNG）と生データ（CSV）が生成されます。

* **Fig J: The Oversampling Paradox (Bar Chart)**
  * 忖度後における「評価1」の真の内訳と、消えたマイノリティの行方を可視化したグラフ。
  * `True Minority in Rating 1`: 忖度に屈しなかった本物のシグナル。
  * `Majority Noise in Rating 1`: マジョリティのゆらぎによって生じた純粋なノイズ。
  * `Missing Signal (Fugitives)`: 評価3や4に逃げ込んでしまった、本来救うべきだったシグナル層。
* **data_J_oversampling_summary.csv**
  * 上記の構成人数を記録した生データ。

## 🚀 実行方法 (Usage)

本コードは **Google Colaboratory** または ローカルのPython環境で実行可能です。

1. `oversampling_sim.py`（または Jupyter Notebook形式）を実行します。
2. 計算完了後、グラフとCSVを格納した `oversampling_paradox_archive.zip` が生成されます。

### ローカル環境での実行に関する注意
ローカルのPython環境（VSCode, JupyterLab等）で実行する場合は、スクリプト内の `from google.colab import files` および、末尾の `files.download(...)` は使用せず、そのままスクリプトを実行してください。カレントディレクトリにZIPファイルが生成されます。

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# スクリプトの実行
python oversampling_sim.py
