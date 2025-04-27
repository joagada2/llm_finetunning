#!/bin/bash
#SBATCH --export=ALL               # inherit your activated env
#SBATCH -A trn040
#SBATCH -J slm-full-pipe
#SBATCH -o .cache/sbatch_logs/%x-%j.out
#SBATCH -e .cache/sbatch_logs/%x-%j.err
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --partition=batch

unset SLURM_EXPORT_ENV

# No conda activation here—job will use your current env

cd /gpfs/wolf2/olcf/trn040/scratch/josephagada/project/llm_finetunning

echo "===== PIPELINE START ====="
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "PWD:  $(pwd)"
echo "Using Python: $(which python3)"
echo "--------------------------"

# STEP 1: Download model & data
echo ">>> Running data preparation"
python3 data_and_model_download.py

echo ">>> Data prep complete at $(date)"
echo "--------------------------"

# STEP 2: Fine-tuning
echo ">>> Launching fine-tuning"
python3 gpu_finetune.py

echo "===== PIPELINE END ====="
echo "Time: $(date)"
