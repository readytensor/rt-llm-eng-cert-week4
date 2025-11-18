"""
evaluate_lora.py
Evaluate a fine-tuned LoRA model on the SAMSum dataset.
Reuses shared utilities for config, dataset loading, and inference.
"""

import os
import json
import torch
from dotenv import load_dotenv
from peft import PeftModel

from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset
from utils.model_utils import setup_model_and_tokenizer
from utils.inference_utils import generate_predictions, compute_rouge
from paths import OUTPUTS_DIR

load_dotenv()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_peft_model(cfg, adapter_dir: str = None, results_dir: str = None):
    """Load base model, attach LoRA adapters, and evaluate."""

    # ----------------------------
    # Model & Tokenizer
    # ----------------------------
    print("Loading base model...")
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=True, use_lora=False, padding_side="left"
    )

    if adapter_dir is None:
        if cfg.get("save_dir", None) is None:
            adapter_dir = os.path.join(OUTPUTS_DIR, "lora_samsum", "lora_adapters")
        else:
            adapter_dir = os.path.join(OUTPUTS_DIR, cfg.get("save_dir"), "lora_adapters")

    else:
        adapter_dir = os.path.join(OUTPUTS_DIR, adapter_dir, "lora_adapters")


    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")

    print(f"Loading fine-tuned LoRA adapters from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ----------------------------
    # Dataset
    # ----------------------------
    print("\nLoading dataset...")
    _, val_data, _ = load_and_prepare_dataset(cfg)
    print(f"Validation set size: {len(val_data)} samples")

    # ----------------------------
    # Inference
    # ----------------------------
    print("\nGenerating summaries...")
    preds = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=val_data,
        task_instruction=cfg["task_instruction"],
        batch_size=cfg.get("eval_batch_size", 4),
    )

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("\nComputing ROUGE metrics...")
    scores = compute_rouge(preds, val_data)

    print("\nEvaluation Results:")
    print(f"  ROUGE-1: {scores['rouge1']:.2%}")
    print(f"  ROUGE-2: {scores['rouge2']:.2%}")
    print(f"  ROUGE-L: {scores['rougeL']:.2%}")

    # ----------------------------
    # Save Outputs
    # ----------------------------
    if results_dir is None:
        results_dir = os.path.join(OUTPUTS_DIR, "lora_samsum")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "eval_results.json")
    preds_path = os.path.join(results_dir, "predictions.jsonl")

    results = {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "num_samples": len(val_data),
        "base_model": cfg["base_model"],
        "adapter_dir": adapter_dir,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(preds_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(preds):
            json.dump(
                {
                    "dialogue": val_data[i]["dialogue"],
                    "reference": val_data[i]["summary"],
                    "prediction": pred,
                },
                f,
            )
            f.write("\n")

    print(f"\nSaved results to {results_path}")
    print(f"Saved predictions to {preds_path}")

    return scores, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    scores, preds = evaluate_peft_model(cfg)

    print("Evaluation complete.")
    print("Sample prediction:\n")
    print(preds[0])


if __name__ == "__main__":
    main()
