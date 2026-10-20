"""DraftFM stage 3 — QLoRA fine-tune of Qwen2.5-VL on the exported factory set.

Input : a directory written by `draftfm export-train` (train.jsonl + images/).
Output: LoRA adapter + merged 16-bit model in --out, ready for GGUF/Ollama
        (see README.md next to this file) and for `draftfm exam --solver ollama`.

Hardware: needs a CUDA GPU with >= 12 GB VRAM for the 3B model at 4-bit
(16 GB comfortable; 24 GB for 7B). The BVES laptop's 4 GB RTX 3050 Ti cannot
run this. First real run 2026-10-17 on a rented RTX 4090: 879 steps, 3h46m,
loss 4.14 -> 0.14. At native A3 resolution (the default since 2026-10-18) an
example is about 2,480 image tokens instead of 1,470, so expect roughly twice
the step time.

**The 10-17 run's model was useless at anything positional** and the cause was
here: images were resized to 1288 px while the coordinates in prompts and
answers stayed in the 1654 px frame they were measured in. Read the guard in
main() before changing --image-width.

    pip install -c <(echo torch==$(python -c "import torch;print(torch.__version__.split('+')[0])")) unsloth unsloth_zoo trl datasets pillow   # never let pip change torch
    python training/finetune_qwen25vl_qlora.py --data D:/draftfm-train --out D:/draftfm-run1
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


#: A prompt or an answer matching any of these carries coordinates in the
#: image's own pixel frame, so the image must be shown at the size those
#: numbers were measured in.
#:
#: ``(pixels`` alone is not enough: ``dimension_value`` says "(pixels, x0 y0
#: x1 y1)" but ``member_schedule`` says only "inside image box [117.7, ...]",
#: and ``text_localisation`` asks for a "pixel bounding box" and answers with
#: ``bbox_px``. All three are pixel-framed; 8,216 of the 14,062 rows in
#: draftfm-train-synth-v1 are.
PIXEL_FRAME_MARKERS = re.compile(r"pixel|bbox_px|image box \[", re.IGNORECASE)


def load_rows(data_dir: Path) -> list[dict]:
    rows = []
    for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        human, gpt = row["conversations"]
        rows.append({
            "image": str(data_dir / row["image"]),
            "prompt": human["value"].replace("<image>\n", "", 1),
            "answer": gpt["value"],
            "task_type": row["task_type"],
        })
    return rows


def pixel_framed(rows: list[dict]) -> list[str]:
    """The task types whose prompt or answer is in the image's pixel frame."""

    return sorted({
        row["task_type"]
        for row in rows
        if PIXEL_FRAME_MARKERS.search(row["prompt"]) or PIXEL_FRAME_MARKERS.search(row["answer"])
    })


def answer_balance(rows: list[dict]) -> dict:
    """Per task: row count, distinct answers, and the share of the commonest.

    Printed before training because a collapsed model is read two ways -- the
    data was skewed, or the model stopped using the input -- and only a count
    separates them. On draftfm-train-synth-v1 the commonest answer of every
    task is under 17%, so the 2026-10-17 run's collapse to one answer per task
    was not skew.
    """

    out = {}
    for task in sorted({row["task_type"] for row in rows}):
        answers = Counter(row["answer"] for row in rows if row["task_type"] == task)
        total = sum(answers.values())
        top, count = answers.most_common(1)[0]
        out[task] = {
            "rows": total,
            "distinct_answers": len(answers),
            "majority_share": round(count / total, 4),
            "majority_answer": top[:80],
        }
    return out


def to_messages(row: dict, image) -> dict:
    return {"messages": [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": row["prompt"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
    ]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4, help="Run 3 default 1e-4 (run 2: 2e-4).")
    ap.add_argument("--vision-layers", action="store_true", help="Also train the vision tower (run 2 did; run 3 does not, to keep the base model's reading).")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0, help="Subsample for a smoke run.")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader workers; 0 = in-process.")
    ap.add_argument("--image-width", type=int, default=0,
                    help="Resize each drawing to this width (aspect kept) before training. 0 (the default) keeps the drawing at "
                         "the size its coordinates were measured in -- A3 at 100 dpi is 1654x1169 px, about 2,480 image tokens "
                         "for Qwen2.5-VL. Refused outright when the data is pixel-framed; see the guard below.")
    ap.add_argument("--max-seq", type=int, default=3072, help="Max tokens per example (image tokens + prompt + answer).")
    a = ap.parse_args()

    from PIL import Image
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    data_dir = Path(a.data)
    rows = load_rows(data_dir)
    if a.max_rows:
        import random
        random.Random(a.seed).shuffle(rows)
        rows = rows[: a.max_rows]
    card = json.loads((data_dir / "dataset_card.json").read_text(encoding="utf-8"))
    print(f"{len(rows)} rows; data classes {card['by_data_class']}; test sources excluded: {card['excluded_test_sources']}")

    # --- the guard -------------------------------------------------------
    # The 2026-10-17 run resized every sheet from 1654 px to 1288 px and left
    # the coordinates in prompts and answers in the 1654 px frame. 383 of the
    # 384 drawings are landscape, so almost the whole epoch was trained with
    # every box about 22% away from the thing it named; the model learned that
    # the numbers in the prompt carry no information and fell back to naming
    # one class per task (exam: dimension_value 17.6% with "vertical" on 86 of
    # 91 items, text_localisation mean IoU 0.0004 with correctly sized boxes in
    # the wrong place). Resizing pixel-framed data is not a tuning knob, so it
    # is refused rather than warned about.
    framed = pixel_framed(rows)
    if a.image_width and framed:
        raise SystemExit(
            f"--image-width {a.image_width} would resize images whose prompts and answers are in the image's own\n"
            f"pixel frame ({', '.join(framed)}). The coordinates would then name the wrong place in every drawing.\n"
            "Train at --image-width 0 (native), or re-export the set with coordinates measured in the width you want."
        )

    balance = answer_balance(rows)
    print("answer balance (a majority share near 1.0 means the task can be won without reading the drawing):")
    for task, b in balance.items():
        print(f"  {task:20s} {b['rows']:6d} rows  {b['distinct_answers']:5d} distinct  "
              f"majority {b['majority_share']:6.1%}  {b['majority_answer']}")

    model, tokenizer = FastVisionModel.from_pretrained(a.base, load_in_4bit=True, use_gradient_checkpointing="unsloth", max_seq_length=a.max_seq)
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=a.vision_layers, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none", random_state=a.seed,
    )
    # Decode each distinct image once, resize it once, and share it across its rows. The set has 14,062 rows
    # but only ~384 distinct drawings; decoding per row held ~82 GB in RAM (OOM-killed, exit 137, 2026-10-17),
    # and the LazyRows torch Dataset that replaced it was rejected by SFTTrainer (TypeError). A plain list of
    # rows sharing ~384 resized images needs about 2 GB.
    images: dict[str, object] = {}

    def image_for(path: str):
        if path not in images:
            im = Image.open(path).convert("RGB")
            if a.image_width and im.width > a.image_width:
                im = im.resize((a.image_width, round(im.height * a.image_width / im.width)), Image.LANCZOS)
            images[path] = im
        return images[path]

    dataset = [to_messages(r, image_for(r["image"])) for r in rows]
    sizes = Counter(im.size for im in images.values())
    print(f"{len(images)} distinct images decoded for {len(dataset)} rows; sizes {sizes.most_common(4)}; max_seq {a.max_seq}")
    if framed:
        print(f"pixel-framed tasks ({', '.join(framed)}) are trained at the drawing's own size -- no resize.")

    # --- the second guard -------------------------------------------------
    # The first guard stops *this script* from resizing. It cannot stop the
    # processor: Qwen2.5-VL's image processor runs its own smart_resize,
    # bounded by min_pixels/max_pixels in preprocessor_config.json, and a
    # build that ships a small max_pixels would reintroduce exactly the
    # 10-17 bug without a line of this file changing. So ask the processor
    # what it actually does to a real sheet and compare.
    if framed and images:
        sample = next(iter(images.values()))
        try:
            grid = tokenizer.image_processor(images=[sample], return_tensors="pt")["image_grid_thw"][0]
            seen_h, seen_w = int(grid[1]) * 14, int(grid[2]) * 14
        except Exception as exc:  # noqa: BLE001 - an unknown processor shape is a warning, not a verdict
            print(f"WARNING: could not ask the processor what it shows the model ({exc}); "
                  "if coordinates look learned-but-scaled, start here")
        else:
            sx, sy = seen_w / sample.width, seen_h / sample.height
            tokens = int(grid[1]) * int(grid[2]) // 4
            print(f"processor shows {sample.width}x{sample.height} as {seen_w}x{seen_h} "
                  f"(scale {sx:.4f} x {sy:.4f}, ~{tokens} image tokens)")
            if abs(sx - 1) > 0.02 or abs(sy - 1) > 0.02:
                raise SystemExit(
                    f"the image processor rescales the drawing by {sx:.3f} x {sy:.3f} before the model sees it,\n"
                    "but the coordinates in prompts and answers are in the drawing's own frame. Raise max_pixels\n"
                    f"in the processor config to at least {sample.width * sample.height} "
                    "(or re-export the set in the frame the processor produces)."
                )
            if tokens + 256 > a.max_seq:
                raise SystemExit(
                    f"{tokens} image tokens plus the prompt will not fit in --max-seq {a.max_seq}; "
                    f"use at least {tokens + 256}."
                )

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        # resize="max" is Unsloth's *no-resize* setting (it sets image_size to
        # None); it is resize="min", the default, that falls back to 512 px
        # when a vision config has no image_size, as Qwen2.5-VL's does not.
        # So the collator was never the thing that shrank the 10-17 run --
        # the --image-width pre-resize above was. Keep "max": None is not an
        # accepted value and raises TypeError.
        model=model, tokenizer=tokenizer, data_collator=UnslothVisionDataCollator(model, tokenizer, resize="max", max_seq_length=a.max_seq),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
            warmup_steps=20, num_train_epochs=a.epochs, learning_rate=a.lr,
            fp16=not is_bf16_supported(), bf16=is_bf16_supported(), logging_steps=10,
            optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine", seed=a.seed,
            output_dir=str(Path(a.out) / "checkpoints"), report_to="none",
            remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=a.max_seq, dataloader_num_workers=a.num_workers,
        ),
    )
    stats = trainer.train()

    # Release everything training needed BEFORE merging. save_pretrained_merged
    # materialises a full fp16 copy of the model, and on a 300-row smoke the
    # cgroup peaked at 29,361 MiB -- 202 MiB under the smallest pod RAM seen
    # (29,563 MiB; other hosts give 58 GiB, and which one you get is luck).
    # The full run decodes 384 drawings rather than 196, about 1.1 GB more,
    # which would put that peak over the limit. The trainer, the dataset and
    # the decoded images are all dead weight by this point.
    import gc

    import torch

    del trainer
    dataset.clear()
    images.clear()
    gc.collect()
    torch.cuda.empty_cache()

    out = Path(a.out)
    model.save_pretrained(out / "lora"); tokenizer.save_pretrained(out / "lora")
    model.save_pretrained_merged(str(out / "merged-16bit"), tokenizer, save_method="merged_16bit")
    (out / "train_run.json").write_text(json.dumps({
        "base": a.base, "rows": len(rows), "image_width": a.image_width, "max_seq": a.max_seq, "epochs": a.epochs, "lr": a.lr, "vision_layers": a.vision_layers, "rank": a.rank,
        # Recorded so a later table can say what the model was shown, not
        # only what it was trained on: this is the field that was wrong on
        # 2026-10-17 and the report did not make it obvious.
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sorted(sizes.items())},
        "pixel_framed_tasks": framed,
        "answer_balance": balance,
        "dataset_card": card, "train_runtime_s": stats.metrics.get("train_runtime"),
        "train_loss": stats.metrics.get("train_loss"),
    }, indent=1), encoding="utf-8")
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
