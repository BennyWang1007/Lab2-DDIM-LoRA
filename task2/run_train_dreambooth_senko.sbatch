#!/bin/bash
#SBATCH --job-name=sd_dreambooth_senko
#SBATCH --partition=devq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --output=/mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_train_dreambooth_senko/%x_%j.out
#SBATCH --error=/mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_train_dreambooth_senko/%x_%j.err

set -euo pipefail

# Ensure output directory exists before writing logs
mkdir -p /mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_train_dreambooth_senko

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Starting DreamBooth LoRA training on Senko-san dataset..."

# Conda activation
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ddpm

# Move to task2 root directory where train_dreambooth_lora.py resides
cd /mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2

# Ensure shell scripts have execute permission
chmod +x scripts/train_dreambooth_lora.sh

# Run the DreamBooth script
bash scripts/train_dreambooth_lora.sh