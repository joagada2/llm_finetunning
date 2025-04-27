#!/bin/bash
#SBATCH -A trn040
#SBATCH -J stablelm-ft
#SBATCH -o .cache/sbatch_logs/%x-%j.out
#SBATCH -e .cache/sbatch_logs/%x-%j.err
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4                # request 4 AMD GPUs
#SBATCH --partition=batch

# Only necessary if you submit via `sbatch --export=NONE`
unset SLURM_EXPORT_ENV

# 1) Load toolchains and Python
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# 2) Activate your environment (must exist on the compute node)
source /gpfs/wolf2/olcf/trn040/world-shared/sajal/env-ft-6.1.3/bin/activate

# 3) Go to your working directory
working_dir="/gpfs/wolf2/olcf/trn040/world-shared/agada/llm_finetunning"
cd "${working_dir}"

# 4) Sanity checks
echo "Running on node:        $(hostname)"
echo "Python3 executable:     $(which python3)"
echo "Current working dir:    $(pwd)"

# 5) Run the fine-tuning
python3 gpu_finetune.py
