torchrun \
  --nnodes $SLURM_JOB_NUM_NODES \
  --nproc_per_node 2 \
  training/train.py \
  --model_name Qwen/Qwen3-1.7B \
  --train_file data/processed/alce_sft.jsonl \
  --output_dir outputs/checkpoints \
  --bf16
