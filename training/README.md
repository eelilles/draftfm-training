# DraftFM stage 3 — training

Stage 2 measured the gap (Qwen2.5-VL 3B: 32.5 % on the frozen test split; reads
values 80–85 %, cannot relate or localise 0–35 %). This directory is the
closing-the-gap loop. Everything up to the GPU step runs on any machine; the
GPU step needs ≥ 12 GB VRAM (not the laptop).

## 1. Export the training set (CPU, seconds)

    python -m bves_drafting.draftfm export-train --out D:\draftfm-train --work-dir D:\draftfm-work --extra-seeds 25:400

2026-10-16: 14 062 rows from 385 synthetic sheets, 15 MB, six task types; the
20 frozen test sources are excluded by id and the split rule, and every row's
prompt is byte-identical to what `draftfm exam` sends. Synthetic-only, so the
directory may go to any GPU, rented or not, with no data-boundary question.
`--include-sanitized` admits BVES sanitised sources — only for a GPU inside the
boundary, and only after the 2026-10-16 leak screen is cleared.

## 2. Fine-tune (GPU)

Normally you do not run this by hand: `tools/rent_and_train.py` in the
`bves-drafting` repository rents the pod, clones this repo at a commit, runs
`run_on_rented_gpu.sh`, fetches the results over ssh and terminates the pod.
By hand it is:

    pip install -c <(echo torch==$(python -c "import torch;print(torch.__version__.split('+')[0])")) unsloth unsloth_zoo trl datasets pillow   # never let pip change torch
    python training/finetune_qwen25vl_qlora.py --data D:\draftfm-train --out D:\draftfm-run1 --max-rows 300   # smoke
    python training/finetune_qwen25vl_qlora.py --data D:\draftfm-train --out D:\draftfm-run1                  # full

3B at 4-bit QLoRA, rank 16, one epoch over 14k image-question pairs.
Measured on a rented RTX 4090: 879 steps at 1288 px took 3h46m. The default
is now **native resolution** (A3 at 100 dpi, 1654x1169 px, ~2,480 image
tokens), so expect roughly 6–7 h.

### Do not resize the drawings

The 2026-10-17 run resized every sheet to 1288 px and left the coordinates in
prompts and answers in the 1654 px frame they were measured in. 383 of the 384
drawings are landscape, so nearly the whole epoch showed every box about 22 %
away from the thing it named. Loss fell normally (4.14 -> 0.14) and the model
came out unable to do anything positional: `"axis": "vertical"` on 86 of 91
dimension items, `"horizontal"` on all 125 grid items, one answer for all 40
stale-reference items, and localisation boxes of the right *size* (166x16 px
against a gold mean of 162x12) in the wrong *place* (slope of model centre on
gold centre: 0.30 in x, 0.05 in y). It is not forgetting and not a scoring
bug: the numbers in the prompt stopped carrying information, so the model
stopped using them.

`--image-width` now defaults to 0 and **refuses** any value when the data is
pixel-framed (8,216 of the 14,062 rows are: `dimension_value`,
`member_schedule`, `text_localisation`). To train smaller, re-export with the
coordinates measured in that width.

## 3. Serve it locally and grade it on the SAME exam

    # GGUF via llama.cpp's convert script, then an Ollama Modelfile:
    python convert_hf_to_gguf.py D:\draftfm-run1\merged-16bit --outfile draftfm-3b.gguf --outtype q8_0
    python convert_hf_to_gguf.py D:\draftfm-run1\merged-16bit --mmproj --outfile draftfm-3b-mmproj.gguf
    ollama create draftfm-3b -f training/Modelfile     # BOTH gguf files beside it; one FROM line loads a blind model
    python -m bves_drafting.draftfm exam --solver ollama --model draftfm-3b --split test --synthetic-only         --cache-dir D:\draftfm-run1\cache --report D:\draftfm-run1\exam.json
    python tools/exam_diff.py D:\draftfm-run1          # accuracy against the majority-class baseline, per task

Compare per task type against `research/draftfm-representation-benchmark.md`
("Stage 2", qwen2.5vl:3b column). The number that matters is `member_schedule`,
`grid_position` and `text_localisation` moving off 35/5/0 % while
`dimension_value`/`title_block_field` hold.

## Where the GPU comes from

| option | VRAM | data boundary | note |
|---|---|---|---|
| workstation bvesws1 | check `nvidia-smi` | inside | if ≥ 12 GB, run step 2 there |
| rented GPU (RunPod RTX 4090, `tools/rent_and_train.py`) | 24 GB | **synthetic set only** | US$0.90/h on demand; ~$6 for a full native-resolution run |
| laptop bves-quentin | 4 GB | inside | export + exam only |

## What this does not do

No BVES drawing is in the training set unless `--include-sanitized` is passed.
Synthetic foundation plans teach the task format and the relate/localise
skills the exam measures; whether that transfers to real BVES sheets is
measured afterwards by running the exam on a local overlay of real bundles
(`draftfm factory` on real DXFs, inside the boundary).
