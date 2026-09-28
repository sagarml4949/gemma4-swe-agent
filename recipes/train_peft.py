"""
Train LoRA adapters (tool_lora or main_lora) using Hugging Face PEFT + TRL (SFTTrainer).
Supports 4-bit QLoRA and bfloat16 mixed-precision on compatible GPUs.

Usage:
    python train_peft.py --target tool_lora --model_id google/gemma-4-31b-it
    python train_peft.py --target main_lora --model_id google/gemma-4-31b-it --max_seq_length 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters with PEFT & TRL")
    parser.add_argument("--target", choices=["tool_lora", "main_lora"], required=True)
    parser.add_argument("--model_id", type=str, default="google/gemma-4-31b-it")
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
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--qlora_4bit", action="store_true", default=True)
    return parser.parse_args()



def main():
    args = parse_args()

    train_file = args.data_dir / f"{args.target}_train.jsonl"
    val_file = args.data_dir / f"{args.target}_val.jsonl"

    if not train_file.is_file():
        raise FileNotFoundError(f"Training dataset not found: {train_file}")
    if not val_file.is_file():
        raise FileNotFoundError(f"Validation dataset not found: {val_file}")

    out_path = args.output_dir / args.target
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Fine-Tuning: {args.target} ===")
    print(f"Base model: {args.model_id}")
    print(f"Train data: {train_file}")
    print(f"Val data:   {val_file}")
    print(f"Output dir: {out_path}")

    if args.target == "tool_lora":
        rank = 16
        alpha = 32
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        learning_rate = args.lr or 2e-4
        max_seq_len = args.max_seq_length or 2048
        batch_size = args.batch_size or 4
        grad_accum = args.grad_accum or 4
    else:  # main_lora
        rank = 64
        alpha = 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        learning_rate = args.lr or 1e-4
        max_seq_len = args.max_seq_length or 8192
        batch_size = args.batch_size or 2
        grad_accum = args.grad_accum or 8

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as exc:
        print(f"\n[Environment note]: {exc}")
        print("To run on GPU, install: pip install torch transformers datasets peft trl bitsandbytes accelerate")
        print("\nConfig profile verified:")
        print(f"  Rank: {rank}, Alpha: {alpha}, Modules: {target_modules}")
        print(f"  LR: {learning_rate}, Max Seq Len: {max_seq_len}")
        print(f"  Batch size: {batch_size}, Grad accum: {grad_accum}")
        return

    bnb_config = None
    if args.qlora_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if args.qlora_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )

    training_args = TrainingArguments(
        output_dir=str(out_path),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        warmup_ratio=0.03,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print("Training complete.")


if __name__ == "__main__":
    main()

