#!/bin/bash
#SBATCH --job-name=sft-alce
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=batch_gpu
#SBATCH --mem=64GB
#SBATCH --output=sft-alce-%j.out
#SBATCH --error=sft-alce-%j.err

uv sync
uv run torchrun \
  --nnodes $SLURM_JOB_NUM_NODES \
  --nproc_per_node 2 \
  training/train.py \
  --model_name Qwen/Qwen3-1.7B \
  --train_file data/processed/dataset.jsonl \
  --output_dir outputs/checkpoints \
  --bf16
