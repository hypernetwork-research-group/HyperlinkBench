# Pipeline Usage Guide

## Overview

HyperlinkBench now includes two pipeline variants:
1. **pipeline** - Original implementation
2. **pipeline_with_batch** - Optimized version with multi-core support and better batching

## Pipeline with Batch

The batched pipeline offers improved performance through:
- Multi-core CPU utilization
- Efficient batch processing
- GPU support when available
- Better memory management with sparse matrices

### Basic Usage

```bash
uv run pipeline_with_batch \
  --dataset_name ARXIV \
  --negative_sampling MotifHypergraphNegativeSampler \
  --hlp_method CommonNeighbors \
  --output_path ./results
```

### All Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_name` | str | **required** | Dataset name: IMDB, COURSERA, ARXIV |
| `--negative_sampling` | str | **required** | Method: SizedHypergraphNegativeSampler, MotifHypergraphNegativeSampler, CliqueHypergraphNegativeSampler |
| `--hlp_method` | str | **required** | Method: CommonNeighbors, NeuralHP, FactorizationMachine |
| `--alpha` | float | 0.5 | First parameter for SizedNegativeSampler |
| `--beta` | int | 1 | Second parameter for SizedNegativeSampler |
| `--output_path` | str | ./results | Directory to save results |
| `--random_seed` | int | None | Seed for reproducibility |
| `--test` | bool | False | Run in test mode (reduced dataset) |
| `--batch_size` | int | 4000 | Training batch size |
| `--num_workers` | int | auto | Number of CPU workers (default: CPU count - 1) |
| `--epochs` | int | 150 | Number of training epochs |

### Examples

**Full training run:**
```bash
uv run pipeline_with_batch \
  --dataset_name ARXIV \
  --negative_sampling MotifHypergraphNegativeSampler \
  --hlp_method CommonNeighbors \
  --batch_size 2000 \
  --num_workers 8 \
  --epochs 100
```

**Quick test with small dataset:**
```bash
uv run pipeline_with_batch \
  --dataset_name ARXIV \
  --negative_sampling MotifHypergraphNegativeSampler \
  --hlp_method CommonNeighbors \
  --test True \
  --epochs 10
```

**With reproducible results:**
```bash
uv run pipeline_with_batch \
  --dataset_name IMDB \
  --negative_sampling SizedHypergraphNegativeSampler \
  --hlp_method NeuralHP \
  --alpha 0.7 \
  --beta 2 \
  --random_seed 42 \
  --epochs 150
```

## Performance Tips

1. **Batch Size**: Adjust based on your GPU memory
   - GPU with 8GB: try 2000-4000
   - GPU with 16GB: try 4000-8000
   - CPU only: try 500-1000

2. **Workers**: Set based on your CPU cores
   - Default uses all cores minus 1
   - Reduce if memory is limited

3. **Test Mode**: Use `--test True` for quick validation before full runs

## Output Files

The pipeline generates:
- **CSV results**: `{output_path}/exp.csv` - Metrics for all runs
- **Confusion matrix**: `{output_path}/{timestamp}_{dataset}_confusion_matrix.png`
- **TensorBoard logs**: `./logs/{timestamp}_{dataset}/` - Training curves

View logs with:
```bash
tensorboard --logdir ./logs
```

## Bug Fixes Applied

### Memory Allocation Error Fix
The negative sampling algorithms (Motif and Clique) now use sparse matrix operations instead of converting to dense matrices, preventing memory allocation errors with large graphs.

**Before:** Tried to allocate petabytes of memory
**After:** Uses sparse operations throughout, handles large graphs efficiently
