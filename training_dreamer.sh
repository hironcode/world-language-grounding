#!/bin/bash

# 必要なパッケージをインストール
pip install jax
pip install -U -r requirements.txt
pip install crafter
pip install ale-py

# 実行時のタイムスタンプを生成
timestamp=$(date +"%Y%m%d_%H%M%S")

# DreamerV3 のメインスクリプトを実行
python dreamerv3/main.py \
  --logdir /workspace/assets/logdir/crafter/${timestamp} \
  --configs crafter \
  --run.train_ratio 32
