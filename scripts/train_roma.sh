#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: train_roma.sh <scenedepict|scanrefer|nr3d|3dllm> <bigru|bert>}"
TEXT_ENCODER="${2:?usage: train_roma.sh <dataset> <bigru|bert>}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/ktj/miniconda3/envs/oneformer3d/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/data}"
VOCAB_PATH="${VOCAB_PATH:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/vocab}"
BERT_PATH="${BERT_PATH:-/home/ktj/Projects/RoMa/pretrained/bert-base-uncased}"
SAVE_PATH="${SAVE_PATH:-checkpoints/roma/${DATASET}/${TEXT_ENCODER}}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
EARLY_STOP_LOG="${EARLY_STOP_LOG:-${SAVE_PATH}/early_stop_monitor.log}"
EARLY_STOP_STATE="${EARLY_STOP_STATE:-${SAVE_PATH}/early_stop.json}"

cd "$(dirname "$0")/.."
read -r EPOCHS LR_UPDATE LR BATCH PRECOMP <<<"$("${PYTHON_BIN}" - "${DATASET}" "${TEXT_ENCODER}" <<'PY'
import sys
from lib.roma_runs import RUN_PRESETS
preset=RUN_PRESETS[(sys.argv[1],sys.argv[2])]
print(preset.epochs,preset.lr_update,preset.learning_rate,preset.batch_size,preset.precomp_enc_type)
PY
)"

EXTRA=()
if [[ "${TEXT_ENCODER}" == "bert" ]]; then
  EXTRA+=(--bert_path "${BERT_PATH}")
  EFFECTIVE_LR="${LEARNING_RATE:-${BERT_LEARNING_RATE:-1e-4}}"
  EFFECTIVE_LR_UPDATE="${LR_UPDATE_OVERRIDE:-${BERT_LR_UPDATE:-5}}"
else
  EFFECTIVE_LR="${LEARNING_RATE:-$LR}"
  EFFECTIVE_LR_UPDATE="${LR_UPDATE_OVERRIDE:-$LR_UPDATE}"
fi
if [[ -s "${SAVE_PATH}/checkpoint.pth" && "${AUTO_RESUME:-0}" == "1" ]]; then
  EXTRA+=(--resume "${SAVE_PATH}/checkpoint.pth")
fi

TRAIN_COMMAND=(
  "${PYTHON_BIN}" train.py
  --data_path "${DATA_ROOT}" --data_root "${DATA_ROOT}" --data_name "${DATASET}"
  --text_enc_type "${TEXT_ENCODER}" --vocab_path "${VOCAB_PATH}"
  --logger_name "${SAVE_PATH}/log" --model_name "${SAVE_PATH}"
  --num_epochs "${NUM_EPOCHS:-$EPOCHS}" --lr_update "${EFFECTIVE_LR_UPDATE}"
  --learning_rate "${EFFECTIVE_LR}" --batch_size "${BATCH_SIZE:-$BATCH}"
  --precomp_enc_type "${PRECOMP_ENC_TYPE:-$PRECOMP}" --workers "${WORKERS:-10}"
  --log_step "${LOG_STEP:-200}" --img_dim 1024 --embed_size 1024 --vse_mean_warmup_epochs 3
  --coding_type VHACoding --alpha 0.1 --pooling_type LSEPooling --belta 0.1
  --drop --criterion ContrastiveLoss --margin 0.05
  "${EXTRA[@]}" "${@:3}"
)

if [[ "${EARLY_STOP_PATIENCE}" =~ ^[0-9]+$ ]] && [[ "${EARLY_STOP_PATIENCE}" -gt 0 ]]; then
  mkdir -p "$(dirname "${EARLY_STOP_LOG}")" "$(dirname "${EARLY_STOP_STATE}")"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${PWD}/../../tools/train_with_early_stop.py" --patience "${EARLY_STOP_PATIENCE}" \
    --log "${EARLY_STOP_LOG}" --state-json "${EARLY_STOP_STATE}" -- "${TRAIN_COMMAND[@]}"
else
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${TRAIN_COMMAND[@]}"
fi
