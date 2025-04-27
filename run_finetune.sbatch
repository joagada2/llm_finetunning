#!/bin/bash
#SBATCH --export=ALL
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

# 1) Go to your project folder
cd /gpfs/wolf2/olcf/trn040/scratch/josephagada/project/llm_finetunning

# 2) Activate project-local venv
#    (assumes you've already run: python3 -m venv venv && venv/bin/pip install -r requirements.txt)
source venv/bin/activate

# 3) Sanity check
echo "===== PIPELINE START ====="
echo "Host:        $(hostname)"
echo "Time:        $(date)"
echo "Workdir:     $(pwd)"
echo "Python exec: $(which python3)"
echo "Packages:"
python3 - <<'EOF'
import sys, torch, transformers, datasets, numpy
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("NumPy:", numpy.__version__)
EOF
echo "--------------------------"

# 4) Download & prepare
echo ">>> Running data preparation"
python3 data_and_model_download.py || { echo "DOWNLOAD FAIL"; exit 1; }

echo ">>> Data prep complete at $(date)"
echo "--------------------------"

# 5) Fine-tuning
echo ">>> Launching fine-tuning"
python3 gpu_finetune.py || { echo "FINETUNE FAIL"; exit 1; }

echo "===== PIPELINE END ====="
echo "Time: $(date)"
