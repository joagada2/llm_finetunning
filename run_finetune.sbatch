#!/bin/bash
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

# 1) Load modules & Python env
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
source /gpfs/wolf2/olcf/trn040/world-shared/sajal/env-ft-6.1.3/bin/activate

# 2) Switch to project directory
cd /gpfs/wolf2/olcf/trn040/scratch/josephagada/project/llm_finetunning

echo "===== PIPELINE START ====="
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "PWD:  $(pwd)"
echo "--------------------------"

# 3) Download & prepare model + data
echo ">>> Running data preparation"
python3 prepare_and_download_finetune.py

echo ">>> Data prep complete at $(date)"
echo "--------------------------"

# 4) Run fine-tuning
echo ">>> Launching fine-tuning"
python3 gpu_finetune_complete.py

echo "===== PIPELINE END ====="
echo "Time: $(date)"
