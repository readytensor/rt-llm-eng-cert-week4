# Week 4: Scaling LLM Training

This repository contains code and examples for Week 4 of the LLM Engineering and Deployment Certification Program, covering memory optimization, distributed training, and production workflows for LLM fine-tuning.

## Topics Covered

- **Distributed Data Parallelism (DDP)** - Speed up training with Accelerate
- **DeepSpeed ZeRO** - Memory-efficient multi-GPU sharding
- **FSDP** - PyTorch's Fully Sharded Data Parallelism
- **Axolotl** - Production-grade training framework
- **Advanced Parallelism** - Tensor and pipeline parallelism concepts

---

## System Requirements

- Multi-GPU machine (RunPod, Lambda Labs, AWS, or local)
- CUDA-compatible GPUs
- Python 3.9+

## Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure environment variables:**

Create a `.env` file in the repository root:

```bash
HF_TOKEN=your-hf-token
HF_USERNAME=your-hf-username
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=your-project-name
WANDB_DISABLED=false
```

**Getting your tokens:**

- **HF_TOKEN**: Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **WANDB_API_KEY**: Find at [wandb.ai/authorize](https://wandb.ai/authorize)

3. **Accept model license:**

For Llama models, accept the license at [meta-llama/Llama-3.2-8B](https://huggingface.co/meta-llama/Llama-3.2-8B)

---

## 1. Distributed Training with Accelerate

Train Llama 3.2 8B with QLoRA across multiple GPUs using Hugging Face Accelerate.

## Training

From the repository root:

```bash
# Baseline QLoRA model
python code/train_qlora_baseline.py


# Single GPU baseline
accelerate launch --config_file code/configs/accelerate_baseline_1gpu.yaml code/train_accelerate.py

# DDP with 2 GPUs
accelerate launch --config_file code/configs/accelerate_ddp_2gpu.yaml code/train_accelerate.py

# DDP with 4 GPUs
accelerate launch --config_file code/configs/accelerate_ddp_4gpu.yaml code/train_accelerate.py

# FSDP with 2 GPUs
accelerate launch --config_file code/configs/accelerate_fsdp_zero2_2gpu.yaml code/train_accelerate.py

# FSDP with 4 GPUs
accelerate launch --config_file code/configs/accelerate_fsdp_zero2_4gpu.yaml code/train_accelerate.py
```

**Outputs saved to:**

- `data/outputs/accelerate_baseline_1gpu/<model_name>-1-gpu/`
- `data/outputs/accelerate_ddp/<model_name>-2-gpus/`
- `data/outputs/accelerate_ddp/<model_name>-4-gpus/`

## Evaluation

After training, evaluate each model (update the model name in the command below). You set it in the `code/config.yaml` file. Use lowercase for the model name.

```bash
# Evaluate baseline QLoRA model
python code/evaluate_model.py  --config code/configs/training/qlora.yaml --model_path data/outputs/baseline_qlora/lora_adapters

# Evaluate DDP with 4 GPUs
python code/evaluate_model.py --model_path data/outputs/accelerate_ddp_4gpu/<model_name>/lora_adapters --config code/configs/training/qlora.yaml
```

**Results saved alongside adapters:**

- `eval_results.json` - ROUGE scores
- `predictions.jsonl` - Model predictions
