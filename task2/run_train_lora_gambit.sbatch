#!/bin/bash
#SBATCH --job-name=sd_lora_train_custom
#SBATCH --partition=devq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --output=/mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_train_lora_custom/%x_%j.out
#SBATCH --error=/mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_train_lora_custom/%x_%j.err

set -euo pipefail

# Ensure output directory exists before writing logs
mkdir -p /mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_train_lora_custom

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Starting LoRA fine-tuning on Gambit_Dataset_and_Lora dataset..."

# Conda activation
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ddpm

# Stay in task2 root directory
cd /mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2

# Run train_lora_custom.sh directly from scripts/ while remaining in task2
chmod +x ./scripts/train_lora_custom.sh
bash ./scripts/train_lora_custom.sh