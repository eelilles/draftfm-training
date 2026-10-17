#!/usr/bin/env bash
# One-shot DraftFM stage 3 on a fresh rented Linux GPU box (RunPod / Lambda / Vast, CUDA 12.x, >= 16 GB VRAM).
# Upload draftfm-train-synth-v1.zip and this repo's training/ folder, then:   bash run_on_rented_gpu.sh
# Produces results/: lora/, merged-16bit/, draftfm-3b.gguf, train_run.json. Download results/ and destroy the box.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-draftfm-train-synth-v1}
if [ ! -d "$DATA" ]; then
  if [ -f "$DATA.zip" ]; then unzip -q "$DATA.zip" -d "$DATA"
  else  # inside a clone of bves-drafting: regenerate the synthetic set from the repo itself (seconds, no upload)
    PYTHONPATH=../src python -m bves_drafting.draftfm export-train --out "$DATA" --work-dir data/work --extra-seeds 25:400
  fi
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q "unsloth[cu124-torch250]" datasets pillow trl
python finetune_qwen25vl_qlora.py --data "$DATA" --out results --max-rows ${SMOKE_ROWS:-0} ${EXTRA_ARGS:-}
# GGUF for Ollama
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python llama.cpp/convert_hf_to_gguf.py results/merged-16bit --outfile results/draftfm-3b.gguf --outtype q8_0
ls -la results
echo "Done. Download results/ then: ollama create draftfm-3b -f Modelfile  (with draftfm-3b.gguf beside it)"
