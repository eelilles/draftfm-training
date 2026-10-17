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
# Keep the image's own torch/CUDA build: no torch pin, no extras (the cu124-torch250 extra downgraded
# torch 2.8->2.5 and broke torchao on the first run, 2026-10-17).
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
# Pin torch/torchvision to the versions already installed so no dependency can swap them, and drop
# packages that a previous attempt may have left pinned to another torch (xformers, torchao).
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
TV_VER=$(python -c "import torchvision; print(torchvision.__version__.split('+')[0])" 2>/dev/null || true)
printf 'torch==%s\n' "$TORCH_VER" > /tmp/draftfm-constraints.txt
[ -n "$TV_VER" ] && printf 'torchvision==%s\n' "$TV_VER" >> /tmp/draftfm-constraints.txt
pip uninstall -y -q xformers torchao 2>/dev/null || true
pip install -q --upgrade --upgrade-strategy only-if-needed -c /tmp/draftfm-constraints.txt unsloth unsloth_zoo trl datasets pillow
python -c "import unsloth, torch; print('unsloth', unsloth.__version__, 'torch', torch.__version__); assert torch.__version__.startswith('$TORCH_VER'), 'torch was changed by pip'"
python finetune_qwen25vl_qlora.py --data "$DATA" --out results --max-rows ${SMOKE_ROWS:-0} ${EXTRA_ARGS:-}
# GGUF for Ollama
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
# Text weights + the vision projector (Qwen2.5-VL needs both for Ollama). Non-fatal: merged-16bit is the
# deliverable; GGUF is a convenience.
python llama.cpp/convert_hf_to_gguf.py results/merged-16bit --outfile results/draftfm-3b.gguf --outtype q8_0 \
  && python llama.cpp/convert_hf_to_gguf.py results/merged-16bit --mmproj --outfile results/draftfm-3b-mmproj.gguf \
  || echo "WARNING: GGUF conversion failed; results/merged-16bit (safetensors) is still complete"
ls -la results
echo "Done. Download results/ then: ollama create draftfm-3b -f Modelfile  (with draftfm-3b.gguf beside it)"
