import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam


def clear_gpu_memory():
    """Clear GPU memory and reset peak memory stats."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_memory_experiment(batch_size, seq_length, model_name="bert-base-uncased"):
    """
    Run a single memory experiment with given batch_size and seq_length.
    
    Returns:
        dict: Contains baseline_mb, peak_mb, and training_mb
    """
    # Clear GPU before starting
    clear_gpu_memory()
    
    # Get baseline
    baseline_mb = torch.cuda.memory_allocated() / 1024**2
    
    try:
        # Load model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name).cuda()
        
        # Create dummy inputs
        inputs = tokenizer(
            ["This is a sample sentence for BERT."] * batch_size,
            max_length=seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        labels = torch.zeros(batch_size, dtype=torch.long).to("cuda")
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=5e-5)
        
        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Optimizer step (allocates optimizer states)
        optimizer.step()
        
        # Measure peak memory
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        training_mb = peak_mb - baseline_mb
        
        # Cleanup
        del model, tokenizer, optimizer, inputs, labels, outputs, loss
        
        return {
            "baseline_mb": baseline_mb,
            "peak_mb": peak_mb,
            "training_mb": training_mb,
            "success": True
        }
        
    except Exception as e:
        return {
            "baseline_mb": baseline_mb,
            "peak_mb": 0,
            "training_mb": 0,
            "success": False,
            "error": str(e)
        }
    finally:
        clear_gpu_memory()


def print_results_table(results):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("GPU MEMORY CONSUMPTION RESULTS")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Seq Length':<12} {'Peak Memory (MB)':<20} {'Training Memory (MB)':<20}")
    print("-" * 80)
    
    for result in results:
        if result["success"]:
            print(f"{result['batch_size']:<12} {result['seq_length']:<12} "
                  f"{result['peak_mb']:<20.2f} {result['training_mb']:<20.2f}")
        else:
            print(f"{result['batch_size']:<12} {result['seq_length']:<12} "
                  f"{'ERROR':<20} {result.get('error', 'Unknown')}")
    
    print("=" * 80)


def main():
    """Run experiments with different configurations."""
    print("=" * 80)
    print("BERT FINE-TUNING: GPU MEMORY ANALYSIS EXPERIMENT")
    print("=" * 80)
    
    # Configuration - 3 + 3 scenarios
    experiments = [
        # Varying batch size (fixed seq_length=256)
        {"batch_size": 8, "seq_length": 256},
        {"batch_size": 16, "seq_length": 256},
        {"batch_size": 32, "seq_length": 256},
        # Varying seq_length (fixed batch_size=16)
        {"batch_size": 16, "seq_length": 128},
        {"batch_size": 16, "seq_length": 256},
        {"batch_size": 16, "seq_length": 512},
    ]
    
    model_name = "bert-base-uncased"
    
    # Run experiments
    results = []
    total_experiments = len(experiments)
    
    for i, config in enumerate(experiments, 1):
        batch_size = config["batch_size"]
        seq_length = config["seq_length"]
        
        print(f"\n[{i}/{total_experiments}] Running experiment: "
              f"batch_size={batch_size}, seq_length={seq_length}")
        
        result = run_memory_experiment(batch_size, seq_length, model_name)
        result["batch_size"] = batch_size
        result["seq_length"] = seq_length
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ Peak memory: {result['peak_mb']:.2f} MB")
            print(f"  ✓ Training memory: {result['training_mb']:.2f} MB")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Print summary table
    print_results_table(results)


if __name__ == "__main__":
    main()