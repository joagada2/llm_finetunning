#!/bin/bash
#SBATCH --export=ALL               # inherit your current env vars
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

# 1) Re-initialize conda (so we can re-activate your env)
module load miniforge3/23.11.0
# point to the same conda you loaded interactively
source /sw/odo/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate env-ft-6.1.3

# 2) Sanity check
echo "===== PIPELINE START ====="
echo "Host:        $(hostname)"
echo "Time:        $(date)"
echo "Workdir:     $(pwd)"
echo "Python exec: $(which python3)"
echo "PYTHONPATH:  $PYTHONPATH"
echo "--------------------------"

# 3) Download & prepare
echo ">>> Running data preparation"
python3 data_and_model_download.py || { echo "DOWNLOAD FAIL"; exit 1; }

echo ">>> Data prep complete at $(date)"
echo "--------------------------"

# 4) Fine-tuning
echo ">>> Launching fine-tuning"
python3 gpu_finetune.py   || { echo "FINETUNE FAIL"; exit 1; }

echo "===== PIPELINE END ====="
echo "Time: $(date)"
