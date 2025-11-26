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

## Usage

### Scenario 1: Baseline QLoRA (Single GPU)

**Training:**

```bash
python code/train_qlora_baseline.py
```

**Outputs:**

- `data/outputs/baseline_qlora/lora_adapters/` - LoRA adapters
- `data/outputs/baseline_qlora/training_duration.json` - Training time

**Evaluation:**

```bash
python code/evaluate_model.py \
    --cfg_path code/configs/training/qlora.yaml \
    --model_path data/outputs/baseline_qlora/lora_adapters
```

**Evaluation outputs:**

- `data/outputs/baseline_qlora/lora_adapters/eval_results.json` - ROUGE scores
- `data/outputs/baseline_qlora/lora_adapters/predictions.jsonl` - Model predictions

---

### Scenario 2: DDP + QLoRA (Multi-GPU)

**Training (2 GPUs):**

```bash
accelerate launch --config_file code/configs/accelerate/ddp_2gpu.yaml code/train_qlora_ddp.py
```

**Training (4 GPUs):**

```bash
accelerate launch --config_file code/configs/accelerate/ddp_4gpu.yaml code/train_qlora_ddp.py
```

**Outputs:**

- `data/outputs/ddp_2gpu/<model-name>/lora_adapters/` - LoRA adapters
- `data/outputs/ddp_2gpu/<model-name>/training_duration.json` - Training time

**Evaluation:**

```bash
# 2 GPU
python code/evaluate_model.py \
    --cfg_path code/configs/training/qlora.yaml \
    --model_path data/outputs/ddp_2gpu/<model-name>/lora_adapters

# 4 GPU
python code/evaluate_model.py \
    --cfg_path code/configs/training/qlora.yaml \
    --model_path data/outputs/ddp_4gpu/<model-name>/lora_adapters
```

**Evaluation outputs:** Same as baseline (in model path directory)

---

### Scenario 3: FSDP (Full Fine-Tuning or LoRA)

FSDP supports 8 combinations: {LoRA, Full FT} × {ZeRO2, ZeRO3} × {2 GPU, 4 GPU}

**Training pattern:**

```bash
accelerate launch --cfg_path code/configs/accelerate/fsdp_<ngpu>gpu_zero<stage>.yaml \
    code/train_fsdp.py --cfg_path code/configs/training/<lora|full_ft>.yaml
```

**Examples:**

```bash
# LoRA with 2 GPU, ZeRO2
accelerate launch --config_file code/configs/accelerate/fsdp_2gpu_zero2.yaml \
    code/train_fsdp.py --cfg_path code/configs/training/lora.yaml

# Full fine-tuning with 4 GPU, ZeRO3
accelerate launch --config_file code/configs/accelerate/fsdp_4gpu_zero3.yaml \
    code/train_fsdp.py --cfg_path code/configs/training/full_ft.yaml
```

**Outputs:**

- `data/outputs/fsdp_<ngpu>gpu_zero<stage>/<model-name>/lora_adapters/` (LoRA)
- `data/outputs/fsdp_<ngpu>gpu_zero<stage>/<model-name>/final_model/` (Full FT)
- `data/outputs/fsdp_<ngpu>gpu_zero<stage>/<model-name>/training_duration.json`

**Evaluation:**

```bash
# LoRA
python code/evaluate_model.py \
    --cfg_path code/configs/training/lora.yaml \
    --model_path data/outputs/fsdp_2gpu_zero2/<model-name>/lora_adapters

# Full FT
python code/evaluate_model.py \
    --cfg_path code/configs/training/full_ft.yaml \
    --model_path data/outputs/fsdp_zero3_4gpu_full/<model-name>/final_model
```

**Evaluation outputs:** Same structure (in model path directory)

---

### Scenario 4: DeepSpeed ZeRO (Full Fine-Tuning or LoRA)

DeepSpeed supports 8 combinations: {LoRA, Full FT} × {ZeRO2, ZeRO3} × {2 GPU, 4 GPU}

**Training pattern:**

```bash
accelerate launch --config_file code/configs/accelerate/deepspeed_<ngpu>gpu_zero<stage>.yaml \
    code/train_deepspeed.py --cfg_path code/configs/training/<lora|full_ft>.yaml
```

**Examples:**

```bash
# LoRA with 2 GPU, ZeRO2
accelerate launch --config_file code/configs/accelerate/deepspeed_2gpu_zero2.yaml \
    code/train_deepspeed.py --cfg_path code/configs/training/lora.yaml

# Full fine-tuning with 4 GPU, ZeRO3
accelerate launch --config_file code/configs/accelerate/deepspeed_4gpu_zero3.yaml \
    code/train_deepspeed.py --cfg_path code/configs/training/full_ft.yaml
```

**Outputs:**

- `data/outputs/deepspeed_zero<stage>_<ngpu>gpu_<lora|full>/<model-name>/lora_adapters/` (LoRA)
- `data/outputs/deepspeed_zero<stage>_<ngpu>gpu_<lora|full>/<model-name>/final_model/` (Full FT)
- `data/outputs/deepspeed_zero<stage>_<ngpu>gpu_<lora|full>/<model-name>/training_duration.json`

**Evaluation:**

```bash
# LoRA
python code/evaluate_model.py \
    --cfg_path code/configs/training/lora.yaml \
    --model_path data/outputs/deepspeed_zero2_2gpu_lora/<model-name>/lora_adapters

# Full FT
python code/evaluate_model.py \
    --cfg_path code/configs/training/full_ft.yaml \
    --model_path data/outputs/deepspeed_zero3_4gpu_full/<model-name>/final_model
```

**Evaluation outputs:** Same structure (in model path directory)
