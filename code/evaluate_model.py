"""
evaluate.py
Evaluate a fine-tuned model (LoRA adapters or full model) on the SAMSum dataset.
Reuses shared utilities for config, dataset loading, and inference.
"""

import os
import json
import argparse
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset
from utils.model_utils import setup_model_and_tokenizer
from utils.inference_utils import generate_predictions, compute_rouge

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Model Detection and Loading
# ---------------------------------------------------------------------------


def detect_model_type(model_path: str):
    """Detect if path contains LoRA adapters or full fine-tuned model."""
    adapter_config = os.path.join(model_path, "adapter_config.json")
    model_config = os.path.join(model_path, "config.json")

    if os.path.exists(adapter_config):
        return "lora_adapters"
    elif os.path.exists(model_config):
        return "full_model"
    else:
        raise ValueError(
            f"Could not detect model type at {model_path}. "
            f"Expected either adapter_config.json (LoRA) or config.json (full model)."
        )


def load_model_for_evaluation(cfg, model_path: str):
    """Load model based on detected type (LoRA adapters or full model)."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = detect_model_type(model_path)

    if model_type == "lora_adapters":
        print(f"\n📦 Detected LoRA adapters at: {model_path}")

        # Read base model from adapter config
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")

        print(f"🔧 Loading base model: {base_model_name}")

        # Load base model with quantization
        model, tokenizer = setup_model_and_tokenizer(
            {"base_model": base_model_name},
            use_4bit=True,
            use_lora=False,
            padding_side="left",
            device_map=device,
        )

        # Load LoRA adapters
        print("🔧 Loading LoRA adapters...")
        model = PeftModel.from_pretrained(model, model_path)
        model.eval()
        print("✅ Successfully loaded LoRA model")

        return model, tokenizer, base_model_name

    else:  # full_model
        print(f"\n📦 Detected full fine-tuned model at: {model_path}")

        # Load tokenizer and model directly from path
        print("🔧 Loading model and tokenizer...")
        
        # If loading from a checkpoint directory where config is a dict, 
        # explicitly load the base model's tokenizer config first or rely on slow tokenizer.
        # The error 'dict object has no attribute model_type' usually happens with fast tokenizers
        # when loading from a directory that doesn't have a full config.json structure expected by AutoTokenizer.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
             # Fallback: Load from base model, then apply tokenizer files from path
            tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
            
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        print("✅ Successfully loaded full fine-tuned model")

        # Extract model name from path for results
        model_name = model_path

        return model, tokenizer, model_name


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(cfg, model_path: str):
    """Load model (LoRA or full) and evaluate on validation set."""

    # Load appropriate model type
    model, tokenizer, model_name = load_model_for_evaluation(cfg, model_path)

    # Results will be saved in the parent directory of the model
    results_dir = os.path.dirname(model_path)

    # ----------------------------
    # Dataset
    # ----------------------------
    print("\n📚 Loading dataset...")
    _, val_data, _ = load_and_prepare_dataset(cfg)
    print(f"Validation set size: {len(val_data)} samples")

    # ----------------------------
    # Inference
    # ----------------------------
    print("\n🔮 Generating summaries...")
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
    print("\n📊 Computing ROUGE metrics...")
    scores = compute_rouge(preds, val_data)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"ROUGE-1: {scores['rouge1']:.4f} ({scores['rouge1']:.2%})")
    print(f"ROUGE-2: {scores['rouge2']:.4f} ({scores['rouge2']:.2%})")
    print(f"ROUGE-L: {scores['rougeL']:.4f} ({scores['rougeL']:.2%})")
    print("=" * 50)

    # ----------------------------
    # Save Outputs
    # ----------------------------
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "eval_results.json")
    preds_path = os.path.join(results_dir, "predictions.jsonl")

    results = {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "num_samples": len(val_data),
        "model_name": model_name,
        "model_path": model_path,
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

    print(f"\n✅ Saved results to {results_path}")
    print(f"✅ Saved predictions to {preds_path}")

    return scores, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model (LoRA adapters or full model) on SAMSum"
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="Path to LoRA adapters OR full fine-tuned model directory",
    )
    parser.add_argument(
        "--cfg_path",
        "-c",
        type=str,
        default=None,
        help="Path to config file (optional)",
    )

    args = parser.parse_args()

    # Load config
    if args.cfg_path:
        cfg = load_config(args.cfg_path)
    else:
        cfg = load_config()

    # Evaluate
    scores, preds = evaluate_model(cfg, args.model_path)

    print("\n✅ Evaluation complete.")
    print("\nSample prediction:")
    print(f"{preds[0][:150]}...")


if __name__ == "__main__":
    main()
