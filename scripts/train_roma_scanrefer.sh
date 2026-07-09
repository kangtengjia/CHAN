#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
DATASET_NAME="scanrefer"
DATA_ROOT="${DATA_ROOT:-/mnt/zhitai2T/ktj/RoMa-data/data}"
VOCAB_PATH="${VOCAB_PATH:-../../RoMa/vocab}"
SAVE_PATH="${SAVE_PATH:-checkpoints/scanrefer_roma_chan_bigru}"

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${GPU_ID}" python ./train.py \
  --data_path="${DATA_ROOT}" --data_root="${DATA_ROOT}" --data_name="${DATASET_NAME}" \
  --text_enc_type=bigru --vocab_path="${VOCAB_PATH}" \
  --logger_name="${SAVE_PATH}/log" --model_name="${SAVE_PATH}" \
  --num_epochs="${NUM_EPOCHS:-300}" --lr_update="${LR_UPDATE:-14}" --learning_rate="${LR:-0.0003}" \
  --precomp_enc_type=selfattention --workers="${WORKERS:-10}" \
  --log_step="${LOG_STEP:-200}" --img_dim=1024 --embed_size=1024 --vse_mean_warmup_epochs=3 \
  --batch_size="${BATCH_SIZE:-8}" \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=LSEPooling --belta=0.1 \
  --drop \
  --criterion=ContrastiveLoss --margin=0.05 \
  "$@"
