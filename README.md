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

    pip install "unsloth[cu124-torch250]" datasets pillow
    python training/finetune_qwen25vl_qlora.py --data D:\draftfm-train --out D:\draftfm-run1 --max-rows 2000   # smoke
    python training/finetune_qwen25vl_qlora.py --data D:\draftfm-train --out D:\draftfm-run1                   # full

3B at 4-bit QLoRA, rank 16, one epoch over 14k image-question pairs: roughly
1–2 h on a 24 GB card. Untested on hardware as of 2026-10-16.

## 3. Serve it locally and grade it on the SAME exam

    # GGUF via llama.cpp's convert script, then an Ollama Modelfile:
    python convert_hf_to_gguf.py D:\draftfm-run1\merged-16bit --outfile draftfm-3b.gguf --outtype q8_0
    ollama create draftfm-3b -f training/Modelfile
    python -m bves_drafting.draftfm exam --solver ollama --model draftfm-3b --split test --report D:\draftfm-run1\exam.json

Compare per task type against `research/draftfm-representation-benchmark.md`
("Stage 2", qwen2.5vl:3b column). The number that matters is `member_schedule`,
`grid_position` and `text_localisation` moving off 35/5/0 % while
`dimension_value`/`title_block_field` hold.

## Where the GPU comes from

| option | VRAM | data boundary | note |
|---|---|---|---|
| workstation bvesws1 | check `nvidia-smi` | inside | if ≥ 12 GB, run step 2 there |
| rented GPU (RunPod / Lambda / Vast, RTX 4090 or A5000 24 GB) | 24 GB | synthetic set only | ~US$0.5–0.8/h, one evening |
| laptop bves-quentin | 4 GB | inside | export + exam only |

## What this does not do

No BVES drawing is in the training set unless `--include-sanitized` is passed.
Synthetic foundation plans teach the task format and the relate/localise
skills the exam measures; whether that transfers to real BVES sheets is
measured afterwards by running the exam on a local overlay of real bundles
(`draftfm factory` on real DXFs, inside the boundary).
