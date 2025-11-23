"""
Fine-tune a Llama 3 model on SAMSum (or another dataset) using LoRA and quantization.
Supports multi-GPU training with Distributed Data Parallelism (DDP) via Accelerate.
Fully integrated with shared utilities and config.yaml.
"""

import os
import json
import time
import argparse
import wandb
import torch
from dotenv import load_dotenv
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from transformers import (
    TrainingArguments,
    Trainer,
)
from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset, build_messages_for_sample
from utils.model_utils import setup_model_and_tokenizer
from paths import OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class PaddingCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch):
        # Convert lists to tensors
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
        attn_masks = [
            torch.tensor(f["attention_mask"], dtype=torch.long) for f in batch
        ]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]

        # Pad to the max length in this batch
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_token_id
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels,
        }


def preprocess_samples(examples, tokenizer, task_instruction, max_length):
    """Tokenize dialogues and apply assistant-only masking for causal LM."""
    input_ids_list, labels_list, attn_masks = [], [], []

    for d, s in zip(examples["dialogue"], examples["summary"]):
        sample = {"dialogue": d, "summary": s}

        # Build chat-style text
        msgs_full = build_messages_for_sample(
            sample, task_instruction, include_assistant=True
        )
        msgs_prompt = build_messages_for_sample(
            sample, task_instruction, include_assistant=False
        )

        text_full = tokenizer.apply_chat_template(
            msgs_full, tokenize=False, add_generation_prompt=False
        )
        text_prompt = tokenizer.apply_chat_template(
            msgs_prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(text_prompt)

        tokens = tokenizer(
            text_full,
            max_length=max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        # Mask non-assistant tokens
        start_idx = len(tokens["input_ids"])
        for i, (start, _) in enumerate(tokens["offset_mapping"]):
            if start >= prompt_len:
                start_idx = i
                break

        labels = [-100] * start_idx + tokens["input_ids"][start_idx:]
        input_ids_list.append(tokens["input_ids"])
        labels_list.append(labels)
        attn_masks.append(tokens["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attn_masks,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(cfg, model, tokenizer, train_data, val_data, save_dir: str):
    """Tokenize datasets, configure Trainer, and run LoRA fine-tuning."""
    task_instruction = cfg["task_instruction"]

    print("\n📚 Tokenizing datasets...")
    tokenized_train = train_data.map(
        lambda e: preprocess_samples(
            e, tokenizer, task_instruction, cfg["sequence_len"]
        ),
        batched=True,
        remove_columns=train_data.column_names,
    )

    tokenized_val = val_data.map(
        lambda e: preprocess_samples(
            e, tokenizer, task_instruction, cfg["sequence_len"]
        ),
        batched=True,
        remove_columns=val_data.column_names,
    )

    collator = PaddingCollator(tokenizer=tokenizer)

    os.makedirs(save_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=cfg["num_epochs"],
        max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 100),
        bf16=cfg.get("bf16", True),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=cfg.get("logging_steps", 25),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to="wandb",
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
    )

    num_gpus = torch.cuda.device_count()
    print(f"\n🎯 Starting LoRA fine-tuning with {num_gpus} GPU(s) using DDP...")

    # Track training duration
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # Calculate duration in minutes
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60.0

    print("\n✅ Training complete!")
    print(
        f"⏱️  Training duration: {duration_minutes:.2f} minutes ({duration_seconds:.2f} seconds)"
    )

    # Save adapters
    adapter_dir = os.path.join(save_dir, "lora_adapters")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"💾 Saved LoRA adapters to {adapter_dir}")

    # Save training duration
    duration_info = {
        "duration_minutes": round(duration_minutes, 3),
        "duration_seconds": round(duration_seconds, 0),
    }
    duration_path = os.path.join(save_dir, "training_duration.json")
    with open(duration_path, "w", encoding="utf-8") as f:
        json.dump(duration_info, f, indent=2)
    print(f"⏱️  Saved training duration to {duration_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path",
        type=str,
        default=None,
        help="Path to training config.yaml",
    )
    args = parser.parse_args()

    # Load training config
    cfg = load_config(args.cfg_path) if args.cfg_path else load_config()

    # Initialize Accelerator for DDP
    accelerator = Accelerator()
    num_gpus = accelerator.num_processes

    # Get model name for folder structure
    model_name = cfg["base_model"].split("/")[-1].lower()

    # Create output directory: outputs/ddp_{n}gpu/{model_name}/
    if num_gpus == 1:
        config_folder = "baseline_1gpu"
    else:
        config_folder = f"ddp_{num_gpus}gpu"

    run_output_dir = os.path.join(OUTPUTS_DIR, config_folder, model_name)
    run_name = f"{config_folder}-{model_name}"

    print(f"\n🔧 Training mode: DDP with {num_gpus} GPU(s)")
    print(f"📁 Output directory: {run_output_dir}")

    # Load dataset
    train_data, val_data, _ = load_and_prepare_dataset(cfg)

    # Setup model with quantization + LoRA for DDP
    model, tokenizer = setup_model_and_tokenizer(
        cfg,
        use_4bit=True,
        use_lora=True,
        padding_side="right",
        device_map=accelerator.local_process_index,
    )

    # Initialize W&B only on main process
    if accelerator.is_main_process:
        wandb.init(
            project=cfg.get("wandb_project", "llama3_samsum"),
            name=run_name,
            config={
                "model": cfg["base_model"],
                "learning_rate": cfg.get("learning_rate", 2e-4),
                "epochs": cfg.get("num_epochs", 1),
                "lora_r": cfg.get("lora_r", 8),
                "lora_alpha": cfg.get("lora_alpha", 16),
                "num_gpus": num_gpus,
                "training_mode": "DDP",
            },
        )

    train_model(cfg, model, tokenizer, train_data, val_data, run_output_dir)

    # Finish W&B run on main process
    if accelerator.is_main_process:
        wandb.finish()

    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    accelerator.free_memory()

    # Properly destroy the process group before exit
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
