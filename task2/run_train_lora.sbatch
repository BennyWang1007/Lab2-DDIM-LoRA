#!/bin/bash
#SBATCH --job-name=sd_lora_train
#SBATCH --partition=devq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --output=/mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_train_lora/%x_%j.out
#SBATCH --error=/mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_train_lora/%x_%j.err

set -euo pipefail

# Ensure output directory exists before writing logs
mkdir -p /mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_train_lora

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"

# echo "Starting LoRA fine-tuning on Senko-san dataset..."
echo "Starting LoRA fine-tuning on prl_crimson_maiden_style dataset..."

# Conda activation
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ddpm

# Move to the script execution directory
cd /mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/scripts

# Grant execution permission and launch training
chmod +x train_lora.sh
bash train_lora.sh