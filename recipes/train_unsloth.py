"""
Unsloth 4-bit QLoRA fine-tuning script for Gemma-4 31B.
Unsloth provides 2-5x faster training and ~70% lower VRAM usage.

Usage:
    python train_unsloth.py --target tool_lora
    python train_unsloth.py --target main_lora --max_seq_length 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters using Unsloth")
    parser.add_argument("--target", choices=["tool_lora", "main_lora"], required=True)
    parser.add_argument("--model_id", type=str, default="unsloth/gemma-4-31b-it")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).parent.parent / "out" / "sft",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "adapters",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train_file = args.data_dir / f"{args.target}_train.jsonl"
    val_file = args.data_dir / f"{args.target}_val.jsonl"

    if not train_file.is_file():
        raise FileNotFoundError(f"Training dataset not found: {train_file}")

    out_path = args.output_dir / args.target
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Unsloth QLoRA Fine-Tuning: {args.target} ===")
    print(f"Base model: {args.model_id}")
    print(f"Dataset:    {train_file}")

    if args.target == "tool_lora":
        rank = 16
        alpha = 32
        max_seq_len = args.max_seq_length or 2048
        lr = 2e-4
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        rank = 64
        alpha = 128
        max_seq_len = args.max_seq_length or 8192
        lr = 1e-4
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as exc:
        print(f"\n[Environment note]: {exc}")
        print("To run with Unsloth, install: pip install unsloth torch trl")
        print("Config summary:")
        print(f"  Rank: {rank}, Alpha: {alpha}, Max Seq Len: {max_seq_len}, LR: {lr}")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=target_modules,
        lora_alpha=alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )

    training_args = TrainingArguments(
        output_dir=str(out_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=lr,
        fp16=False,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=max_seq_len,
        dataset_text_field="messages",
        args=training_args,
    )

    trainer.train()
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
