"""DraftFM stage 3 — QLoRA fine-tune of Qwen2.5-VL on the exported factory set.

Input : a directory written by `draftfm export-train` (train.jsonl + images/).
Output: LoRA adapter + merged 16-bit model in --out, ready for GGUF/Ollama
        (see README.md next to this file) and for `draftfm exam --solver ollama`.

Hardware: needs a CUDA GPU with >= 12 GB VRAM for the 3B model at 4-bit
(16 GB comfortable; 24 GB for 7B). The BVES laptop's 4 GB RTX 3050 Ti cannot run
this. UNTESTED on real hardware as of 2026-10-16 -- it is the standard Unsloth
recipe with DraftFM's data contract wired in; the first run is the test.

    pip install "unsloth[cu124-torch250]" datasets pillow
    python training/finetune_qwen25vl_qlora.py --data D:/draftfm-train --out D:/draftfm-run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0, help="Subsample for a smoke run.")
    ap.add_argument("--seed", type=int, default=3407)
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

    model, tokenizer = FastVisionModel.from_pretrained(a.base, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none", random_state=a.seed,
    )
    dataset = [to_messages(r, Image.open(r["image"]).convert("RGB")) for r in rows]

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
            warmup_steps=20, num_train_epochs=a.epochs, learning_rate=a.lr,
            fp16=not is_bf16_supported(), bf16=is_bf16_supported(), logging_steps=10,
            optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine", seed=a.seed,
            output_dir=str(Path(a.out) / "checkpoints"), report_to="none",
            remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
        ),
    )
    stats = trainer.train()
    out = Path(a.out)
    model.save_pretrained(out / "lora"); tokenizer.save_pretrained(out / "lora")
    model.save_pretrained_merged(str(out / "merged-16bit"), tokenizer, save_method="merged_16bit")
    (out / "train_run.json").write_text(json.dumps({
        "base": a.base, "rows": len(rows), "epochs": a.epochs, "lr": a.lr, "rank": a.rank,
        "dataset_card": card, "train_runtime_s": stats.metrics.get("train_runtime"),
        "train_loss": stats.metrics.get("train_loss"),
    }, indent=1), encoding="utf-8")
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
