#!/bin/bash
#SBATCH --export=ALL
#SBATCH -A trn040
#SBATCH -J slm-finetune
#SBATCH -o .cache/sbatch_logs/%x-%j.out
#SBATCH -e .cache/sbatch_logs/%x-%j.err
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --partition=batch

unset SLURM_EXPORT_ENV

# 1) Go to your project folder
cd /gpfs/wolf2/olcf/trn040/scratch/josephagada/project/llm_finetunning

# 2) Activate project-local venv
source venv/bin/activate

# 3) Sanity check
echo "===== FINE-TUNE START ====="
echo "Host:        $(hostname)"
echo "Time:        $(date)"
echo "Workdir:     $(pwd)"
echo "Python exec: $(which python3)"
echo "--------------------------"

# 4) Fine-tuning only
echo ">>> Launching fine-tuning"
python3 gpu_finetune.py || { echo "FINETUNE FAIL"; exit 1; }

echo "===== FINE-TUNE END ====="
echo "Time: $(date)"
