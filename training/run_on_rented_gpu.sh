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
  # python's zipfile, because `unzip` is not in every CUDA base image and a
  # run that dies on a missing unzip still costs a pod.
  if [ -f "$DATA.zip" ]; then python -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$DATA.zip" "$DATA"
  else  # inside a clone of bves-drafting: regenerate the synthetic set from the repo itself (seconds, no upload)
    PYTHONPATH=../src python -m bves_drafting.draftfm export-train --out "$DATA" --work-dir data/work --extra-seeds 25:400
  fi
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
# Ubuntu 24.04 marks its system Python externally managed (PEP 668) and pip refuses to install
# into it. That interpreter is where this image's torch lives -- `python -c "import torch"` below
# proves it -- so installing beside torch is the correct target, not a mistake to work around with
# a venv that would not see it. A rented container is disposable; there is no system to protect.
# Non-interactive ssh does not read ~/.bashrc, which is why a hand-driven session never hit this
# (2026-10-18, first unattended run: exit 1 at the unsloth install).
export PIP_BREAK_SYSTEM_PACKAGES=1
# Keep the image's own torch/CUDA build: no torch pin, no extras (the cu124-torch250 extra downgraded
# torch 2.8->2.5 and broke torchao on the first run, 2026-10-17).
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
# Pin torch/torchvision to the versions already installed so no dependency can swap them, and drop
# packages that a previous attempt may have left pinned to another torch (xformers, torchao).
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
TV_VER=$(python -c "import torchvision; print(torchvision.__version__.split('+')[0])" 2>/dev/null || true)
printf 'torch==%s\n' "$TORCH_VER" > /tmp/draftfm-constraints.txt
if [ -n "$TV_VER" ]; then printf 'torchvision==%s\n' "$TV_VER" >> /tmp/draftfm-constraints.txt; fi
pip uninstall -y -q xformers torchao 2>/dev/null || true
pip install -q --upgrade --upgrade-strategy only-if-needed -c /tmp/draftfm-constraints.txt unsloth unsloth_zoo trl datasets pillow
python -c "import unsloth, torch; print('unsloth', unsloth.__version__, 'torch', torch.__version__); assert torch.__version__.startswith('$TORCH_VER'), 'torch was changed by pip'"
# Inductor compile workers. TORCHINDUCTOR_COMPILE_THREADS=1 alone did NOT stop torch 2.8 from
# forking ~30 of them (the container sees nproc=128 on an 8-vCPU pod, and the env var was read
# after the pool was sized); 32 workers forked from a 7.5 GB parent OOM-killed the full run on a
# 58 GB cgroup, 2026-10-17. Belt and braces: the variable, torch._inductor.config set in-process
# before the first compile (sitecustomize, so it applies however python is entered), and taskset
# to the CPUs the pod actually has.
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cat > "$PWD/sitecustomize.py" <<'PYEOF'
try:
    import torch._inductor.config as _c
    _c.compile_threads = 1
except Exception:
    pass
PYEOF
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# Every one of these is written as an if, not as `test && assign`. Under
# `set -e` a bare `[ ... ] && x=y` whose test is false IS a failing statement
# and aborts the script -- which is exactly how the 2026-10-18 10:00 run died,
# silently, right after the unsloth install and before the first echo.
CPUS=$(nproc --all 2>/dev/null || echo 8)
QUOTA=$( (cat /sys/fs/cgroup/cpu.max 2>/dev/null || true) | awk '{if ($1!="max") print int($1/$2)}' || true)
if [ -n "${QUOTA:-}" ] && [ "${QUOTA}" -gt 0 ] 2>/dev/null; then CPUS=$QUOTA; fi
echo "limiting compile workers to 1; cpus=$CPUS"

# Memory monitor: the first OOM kill (exit 137) left no record of how close the run had been.
# cgroup v2 then v1, whichever the pod has.
# anon= is the number that decides an OOM. memory.current also counts page
# cache, which the kernel reclaims instead of killing anything -- writing
# 7 GB of safetensors and tarring 4.8 GB of GGUF pushed memory.current to
# 29 GiB on a run whose anonymous memory was a fraction of that, and reading
# that as "202 MiB from an OOM" was wrong (2026-10-18). Report both.
( while true; do
    USED=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
    LIM=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 0)
    ANON=$(awk '/^anon /{print $2; found=1} END{if(!found) print ""}' /sys/fs/cgroup/memory.stat 2>/dev/null)
    if [ -z "${ANON:-}" ]; then ANON=$(awk '/^rss /{print $2}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null); fi
    echo "[mem] $(date -u +%H:%M:%S) anon=$(( ${ANON:-0} /1024/1024))MiB current=$((USED/1024/1024))MiB limit=$((LIM/1024/1024))MiB procs=$(ls /proc | grep -c '^[0-9]')"
    sleep 30
  done ) &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null || true' EXIT

# taskset is a convenience, not a requirement: a container whose cpuset is not 0..N-1 makes it
# fail outright, and losing the whole run to a CPU-pinning nicety would be absurd. Probe it.
PIN=""
if command -v taskset >/dev/null 2>&1 && taskset -c 0-$((CPUS-1)) true 2>/dev/null; then
  PIN="taskset -c 0-$((CPUS-1))"
else
  echo "taskset unavailable or cpuset is not 0-$((CPUS-1)); relying on the env vars alone"
fi
echo "=== training starts ==="
$PIN python finetune_qwen25vl_qlora.py --data "$DATA" --out "$OUT" --max-rows ${SMOKE_ROWS:-0} ${EXTRA_ARGS:-}
echo "=== training done ==="
# GGUF for Ollama
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
# llama.cpp's requirements-convert_hf_to_gguf.txt pins torch~=2.11, so installing it under the
# constraints file (torch==2.8.0) is unsatisfiable and `set -e` aborted the 10-17 run here --
# after the merge, before GGUF, SHA256SUMS and the handoff. Installing it UNconstrained is worse:
# it replaced torch 2.8.0+cu128 with 2.11.0+cpu and poisoned the pod for the next run. The
# converter needs none of that torch: gguf-py plus three tokeniser packages is the whole of it.
pip install -q -c /tmp/draftfm-constraints.txt ./llama.cpp/gguf-py sentencepiece protobuf mistral-common
python -c "import torch; assert torch.__version__.startswith('$TORCH_VER'), 'torch was changed by the GGUF install'"
# Text weights + the vision projector (Qwen2.5-VL needs both for Ollama). Non-fatal: merged-16bit is the
# deliverable; GGUF is a convenience.
python llama.cpp/convert_hf_to_gguf.py "$OUT"/merged-16bit --outfile "$OUT"/draftfm-3b.gguf --outtype q8_0 \
  && python llama.cpp/convert_hf_to_gguf.py "$OUT"/merged-16bit --mmproj --outfile "$OUT"/draftfm-3b-mmproj.gguf \
  || echo "WARNING: GGUF conversion failed; "$OUT"/merged-16bit (safetensors) is still complete"
ls -la "$OUT"
( cd "$OUT" && sha256sum draftfm-3b.gguf draftfm-3b-mmproj.gguf train_run.json 2>/dev/null | tee SHA256SUMS )

# Pack everything the laptop needs into one file and checksum the file itself, so a truncated
# transfer is caught by the fetcher rather than by `ollama create` an hour later.
( cd "$OUT" \
  && tar cf draftfm-results.tar draftfm-3b.gguf draftfm-3b-mmproj.gguf train_run.json SHA256SUMS lora \
  && sha256sum draftfm-results.tar > draftfm-results.tar.sha256 \
  && ls -la draftfm-results.tar )

# tools/rent_and_train.py in bves-drafting watches the log for this exact line and then scps the
# tar off the pod. Keep the text stable.
echo "=== RESULTS READY ==="
echo "$OUT/draftfm-results.tar"

# `runpodctl send` prints a one-time code and WAITS for a human to run `runpodctl receive <code>`,
# so it cannot be part of an unattended run. Off by default; AUTO_SEND=1 brings it back for a
# hand-driven session.
if [ "${AUTO_SEND:-0}" = "1" ] && command -v runpodctl >/dev/null 2>&1; then
  ( cd "$OUT" && runpodctl send draftfm-results.tar )
fi
echo "Done. Fetch $OUT/draftfm-results.tar, then: ollama create draftfm-3b -f Modelfile"
