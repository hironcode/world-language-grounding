#!/bin/bash

# 必要なパッケージをインストール
pip install jax
pip install -U -r requirements.txt
pip install crafter
pip install ale-py

# cp dreamerv3_edits/config.yaml dreamerv3/dreamerv3
# cp dreamerv3_edits/crafter.py dreamerv3/embodied/envs/

# 実行時のタイムスタンプを生成
timestamp=$(date +"%Y%m%d_%H%M%S")

# DreamerV3 のメインスクリプトを実行
python dreamerv3/main.py \
  --logdir ~/logdir/${timestamp} \
  --configs crafter \
  --run.train_ratio 32 \
  --env.crafter.use_logdir True \
  --env.crafter.use_seed True

# 継続学習したい場合、以下を実行
# python dreamerv3/main.py \
#   #　例： --logdir ~/logdir/20250101T000000 \
#   --logdir ~/logdir/path/to/existing/logdir/timestamp \
#   --configs crafter \
#   --run.train_ratio 32