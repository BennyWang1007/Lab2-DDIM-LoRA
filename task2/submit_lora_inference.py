#!/usr/bin/env python3
"""Submit batch seed generation experiments (both LoRA and baseline) as independent Slurm jobs.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("/mnt/bapve/thome/wangyuhao/VideoGenerations/Lab2-DDIM-LoRA/task2/results/run_eval_seeds")

# Slurm resource specifications matching your requested cluster configuration
SBATCH_OPTIONS = {
    "partition": "devq",
    "qos": "normal",
    "nodes": "1",
    "ntasks": "1",
    "cpus-per-task": "8",
    "mem": "32G",
    "time": "04:00:00",
    "gres": "gpu:nvidia_geforce_rtx_4090:1",
}

RUNS_BASE = Path("/mnt/bapve/shome/wangyuhao/runs")

# Configure the LoRA models you want to evaluate
EXPERIMENTS: list[dict[str, str | int]] = [
    {
        "name": "dreambooth_senko_san_drinking_tea",
        "lora_path": str(RUNS_BASE / "dreambooth_senko_san" / "pytorch_lora_weights.safetensors"),
        "prompt": "sks senko san drinking tea",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "dreambooth_senko_san_smiling",
        "lora_path": str(RUNS_BASE / "dreambooth_senko_san" / "pytorch_lora_weights.safetensors"),
        "prompt": "a photo of senko san smiling",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "dreambooth_senko_san_training-like_prompt",
        "lora_path": str(RUNS_BASE / "dreambooth_senko_san" / "pytorch_lora_weights.safetensors"),
        "prompt": "senko_san, 1girl, solo, looking at viewer, outdoors",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "dreambooth_senko_san_cat",
        "lora_path": str(RUNS_BASE / "dreambooth_senko_san" / "pytorch_lora_weights.safetensors"),
        "prompt": "a photo of a cat",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_gambit_lora_general",
        "lora_path": str(RUNS_BASE / "sd-Gambit_Dataset_and_Lora-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "a photo of gambit playing cards",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_gambit_lora_training-like_prompt",
        "lora_path": str(RUNS_BASE / "sd-Gambit_Dataset_and_Lora-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "solo, 1boy, male_focus",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_gambit_lora_cat",
        "lora_path": str(RUNS_BASE / "sd-Gambit_Dataset_and_Lora-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "a photo of a cat",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_crimson_maiden_lora_general",
        "lora_path": str(RUNS_BASE / "sd-prl_crimson_maiden_style-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "crimson maiden style portrait",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_crimson_maiden_lora_general2",
        "lora_path": str(RUNS_BASE / "sd-prl_crimson_maiden_style-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "a painting of a woman in crimson maiden style",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_crimson_maiden_lora_cat",
        "lora_path": str(RUNS_BASE / "sd-prl_crimson_maiden_style-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "a painting of a cat",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_crimson_maiden_lora_training-like",
        "lora_path": str(RUNS_BASE / "sd-prl_crimson_maiden_style-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "a painting of a woman, prl_crimson_maiden_style",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_senko_san_lora_drinking_tea",
        "lora_path": str(RUNS_BASE / "sd-Senko-san-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "sks senko san drinking tea",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_senko_san_lora_smiling",
        "lora_path": str(RUNS_BASE / "sd-Senko-san-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "a photo of senko san smiling",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_senko_san_lora_training-like_prompt",
        "lora_path": str(RUNS_BASE / "sd-Senko-san-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "senko_san, 1girl, solo, looking at viewer, outdoors",
        "start_seed": 1,
        "end_seed": 200,
    },
    {
        "name": "sd_senko_san_lora_cat",
        "lora_path": str(RUNS_BASE / "sd-Senko-san-model-lora" / "pytorch_lora_weights.safetensors"),
        "prompt": "a photo of a cat",
        "start_seed": 1,
        "end_seed": 200,
    },
]


def make_job_script(exp: dict[str, str | int], exp_output_dir: Path, use_lora: bool = True) -> str:
    name = exp["name"]
    lora_path = exp["lora_path"]
    prompt = exp["prompt"]
    start_seed = exp["start_seed"]
    end_seed = exp["end_seed"]
    job_name = f"eval_{name}"

    directives = "\n".join(
        f"#SBATCH --{key}={value}" for key, value in SBATCH_OPTIONS.items()
    )

    log_dir = exp_output_dir / "logs"
    if use_lora:
        img_dir = exp_output_dir / "images"
    else:
        img_dir = exp_output_dir / "images_baseline"

    # Conditionally add the lora_path argument if specified
    lora_arg = f"    --lora_path {shlex.quote(str(lora_path))} \\\n" if use_lora else ""

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
{directives}
#SBATCH --output={log_dir}/slurm-%j.out.log
#SBATCH --error={log_dir}/slurm-%j.err.log

set -euo pipefail

REPO_ROOT={shlex.quote(str(REPO_ROOT))}
cd "$REPO_ROOT"

echo "Job ID: ${{SLURM_JOB_ID:-manual}}"
echo "Node: $(hostname)"
echo "Experiment: {name}"
echo "Seed range: {start_seed} -> {end_seed}"

# Initialize conda
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ddpm

# Run the inference generation script
python generate_lora_images.py \\
    --model_name "CompVis/stable-diffusion-v1-4" \\
{lora_arg}    --prompt {shlex.quote(str(prompt))} \\
    --start_seed {start_seed} \\
    --end_seed {end_seed} \\
    --num_inference_steps 30 \\
    --guidance_scale 7.5 \\
    --output_dir {shlex.quote(str(img_dir))}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts without calling sbatch.",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Specify a single experiment name to run (e.g., 'dreambooth_senko_san'). Runs all if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts_dir = OUTPUT_ROOT / "run_submit_lora_inference"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    experiments_to_run = EXPERIMENTS
    if args.exp:
        experiments_to_run = [e for e in EXPERIMENTS if e["name"] == args.exp]
        if not experiments_to_run:
            raise ValueError(f"Experiment '{args.exp}' not found in configuration list.")

    for exp in experiments_to_run:
        name = exp["name"]
        assert isinstance(name, str), f"Experiment name must be a string, got {type(name)}"
        exp_output_dir = OUTPUT_ROOT / name
        (exp_output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (exp_output_dir / "images").mkdir(parents=True, exist_ok=True)
        (exp_output_dir / "images_baseline").mkdir(parents=True, exist_ok=True)

        job_file = scripts_dir / f"eval_{name}.sh"
        job_content = make_job_script(exp, exp_output_dir, use_lora=True)
        job_file.write_text(job_content, encoding="utf-8")
        job_file.chmod(0o755)

        if args.dry_run:
            print(f"[dry-run] Generated script: {job_file}")
            continue

        result = subprocess.run(
            ["sbatch", str(job_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Submitted {name}: {result.stdout.strip()}")

        job_file = scripts_dir / f"eval_{name}_baseline.sh"
        job_content = make_job_script(exp, exp_output_dir, use_lora=False)
        job_file.write_text(job_content, encoding="utf-8")
        job_file.chmod(0o755)

        if args.dry_run:
            print(f"[dry-run] Generated baseline script: {job_file}")
            continue

        result = subprocess.run(
            ["sbatch", str(job_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Submitted baseline {name}: {result.stdout.strip()}")


if __name__ == "__main__":
    main()
