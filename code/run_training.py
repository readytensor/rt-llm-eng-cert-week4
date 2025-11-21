#!/usr/bin/env python3
"""
Interactive training launcher for Week 4 distributed training scenarios.
"""
import os
import sys
import subprocess
import inquirer


def main():
    print("\n" + "=" * 60)
    print("🚀 LLM Training Launcher - Week 4")
    print("=" * 60 + "\n")

    # Question 1: Single or Multiple GPUs
    questions = [
        inquirer.List(
            "gpu_count",
            message="How many GPUs do you want to use?",
            choices=["Single GPU (Baseline)", "Multiple GPUs"],
        ),
    ]
    answers = inquirer.prompt(questions)

    if answers["gpu_count"] == "Single GPU (Baseline)":
        # Single GPU - just run baseline
        script = "code/train_qlora_baseline.py"
        config = "code/configs/training/qlora.yaml"

        print(f"\n📋 Running: {script}")
        print(f"📄 Config: {config}\n")

        cmd = f"python {script} --config {config}"

        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with exit code {e.returncode}")
            sys.exit(e.returncode)

    else:
        # Multiple GPUs - ask for strategy
        questions = [
            inquirer.List(
                "strategy",
                message="Which distributed training strategy?",
                choices=["DDP (Data Parallel)", "FSDP (Fully Sharded)", "DeepSpeed"],
            ),
            inquirer.List(
                "num_gpus",
                message="How many GPUs?",
                choices=["2", "4"],
            ),
        ]
        answers.update(inquirer.prompt(questions))

        # Map choices to configs and scripts
        strategy_map = {
            "DDP (Data Parallel)": {
                "script": "code/train_qlora_ddp.py",
                "config": "code/configs/training/qlora.yaml",
                "prefix": "ddp",
            },
            "FSDP (Fully Sharded)": {
                "script": "code/train_fsdp.py",
                "config": "code/configs/training/full_ft.yaml",
                "prefix": "fsdp_zero2",
            },
            "DeepSpeed": {
                "script": "code/train_deepspeed.py",
                "config": "code/configs/training/full_ft.yaml",
                "prefix": "deepspeed_zero2",
            },
        }

        strategy_info = strategy_map[answers["strategy"]]
        num_gpus = answers["num_gpus"]

        accelerate_config = (
            f"code/configs/accelerate/{strategy_info['prefix']}_{num_gpus}gpu.yaml"
        )
        script = strategy_info["script"]
        train_config = strategy_info["config"]

        # Validate files exist
        if not os.path.exists(accelerate_config):
            print(f"❌ Error: Accelerate config not found: {accelerate_config}")
            sys.exit(1)
        if not os.path.exists(script):
            print(f"❌ Error: Training script not found: {script}")
            sys.exit(1)
        if not os.path.exists(train_config):
            print(f"❌ Error: Training config not found: {train_config}")
            sys.exit(1)

        print(f"\n📋 Strategy: {answers['strategy']}")
        print(f"🔧 Accelerate config: {accelerate_config}")
        print(f"📄 Training config: {train_config}")
        print(f"🐍 Script: {script}\n")

        # Construct and run command
        cmd = f"accelerate launch --config_file {accelerate_config} {script} --config {train_config}"

        print(f"🚀 Running command:\n{cmd}\n")

        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with exit code {e.returncode}")
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()
