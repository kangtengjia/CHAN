#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/ktj/miniconda3/envs/oneformer3d/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/data}"
BERT_PATH="${BERT_PATH:-/home/ktj/Projects/RoMa/pretrained/bert-base-uncased}"
cd "$(dirname "$0")/.."
"${PYTHON_BIN}" - "${DATA_ROOT}" "${BERT_PATH}" <<'PY'
import json,sys
from lib.datasets.roma import load_roma_bundle
from lib.roma_config import require_bert_path
root,bert=sys.argv[1:]
require_bert_path(bert)
out={}
for dataset in ['scenedepict','scanrefer','nr3d','3dllm']:
 out[dataset]={}
 for split in ['train','val']:
  bundle=load_roma_bundle(root,dataset,split)
  out[dataset][split]={'captions':len(bundle.captions),'scenes':len(bundle.feature_scene_ids),'shape':list(bundle.features.shape)}
print(json.dumps(out,indent=2))
PY
