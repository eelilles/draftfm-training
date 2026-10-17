#!/usr/bin/env bash
# One-shot DraftFM stage 3 on a fresh rented Linux GPU box (RunPod / Lambda / Vast, CUDA 12.x, >= 16 GB VRAM).
# Upload draftfm-train-synth-v1.zip and this repo's training/ folder, then:   bash run_on_rented_gpu.sh
# Produces results/: lora/, merged-16bit/, draftfm-3b.gguf, train_run.json. Download $OUT/ and destroy the box.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-draftfm-train-synth-v1}
OUT=${OUT:-results}          # put it on the persistent volume (e.g. OUT=/workspace/results-full) on RunPod
mkdir -p "$OUT"
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
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}  # 32 Inductor workers forked from a 7.5 GB parent OOM-killed the full run on a 58 GB cgroup, 2026-10-17
python finetune_qwen25vl_qlora.py --data "$DATA" --out "$OUT" --max-rows ${SMOKE_ROWS:-0} ${EXTRA_ARGS:-}
# GGUF for Ollama
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -c /tmp/draftfm-constraints.txt -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt  # unconstrained, this pulled CPU torch 2.11 over cu128 after the GGUF export and poisoned the next run, 2026-10-17
# Text weights + the vision projector (Qwen2.5-VL needs both for Ollama). Non-fatal: merged-16bit is the
# deliverable; GGUF is a convenience.
python llama.cpp/convert_hf_to_gguf.py "$OUT"/merged-16bit --outfile "$OUT"/draftfm-3b.gguf --outtype q8_0 \
  && python llama.cpp/convert_hf_to_gguf.py "$OUT"/merged-16bit --mmproj --outfile "$OUT"/draftfm-3b-mmproj.gguf \
  || echo "WARNING: GGUF conversion failed; "$OUT"/merged-16bit (safetensors) is still complete"
ls -la "$OUT"
( cd "$OUT" && sha256sum draftfm-3b.gguf draftfm-3b-mmproj.gguf train_run.json 2>/dev/null | tee SHA256SUMS )
echo "Done. Download $OUT/ then: ollama create draftfm-3b -f Modelfile  (with draftfm-3b.gguf beside it)"
# Hand the results off before anyone can terminate the box: pack them and offer them with runpodctl.
# `runpodctl send` prints a one-time code and WAITS until `runpodctl receive <code>` runs on the laptop,
# so this line stays in the log until the transfer completes. Set AUTO_SEND=0 to skip.
if [ "${AUTO_SEND:-1}" = "1" ] && command -v runpodctl >/dev/null 2>&1; then
  ( cd "$OUT" && tar cf draftfm-results.tar draftfm-3b.gguf draftfm-3b-mmproj.gguf train_run.json SHA256SUMS lora && ls -la draftfm-results.tar )
  echo "=== RESULTS READY: on the laptop run the 'runpodctl receive ...' line printed below ==="
  ( cd "$OUT" && runpodctl send draftfm-results.tar )
fi
