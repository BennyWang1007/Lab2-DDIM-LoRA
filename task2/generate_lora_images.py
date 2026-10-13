#!/usr/bin/env python3
import os
import argparse
import torch
from diffusers import StableDiffusionPipeline

# Attempt import from utils, provide fallback if executed standalone
try:
    from utils import seed_everything
except ImportError:
    import random
    import numpy as np

    def seed_everything(seed: int = 2024) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch image generation with optional LoRA across seed range.")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="Base model checkpoint or Hugging Face ID.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights (.safetensors). If omitted/empty, runs baseline model.")
    parser.add_argument("--prompt", type=str, default="a photo of a cat", help="Text prompt for generation.")
    parser.add_argument("--start_seed", type=int, default=1, help="Starting seed (inclusive).")
    parser.add_argument("--end_seed", type=int, default=100, help="Ending seed (inclusive).")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on.")
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="Directory where generated images will be saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading base model: {args.model_name}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    )

    if args.lora_path and args.lora_path.lower() != "none":
        print(f"[INFO] Loading LoRA weights from: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)
    else:
        print("[INFO] No LoRA path provided. Running baseline model generation.")

    pipe = pipe.to(args.device)

    print(f"[INFO] Starting inference for seeds {args.start_seed} to {args.end_seed}...")
    for seed in range(args.start_seed, args.end_seed + 1):
        filename = f"seed_{seed:04d}.png"
        filepath = os.path.join(args.output_dir, filename)

        if os.path.exists(filepath):
            print(f"[INFO] Seed {seed} already exists at {filepath}. Skipping generation.")
            continue

        seed_everything(seed)

        image = pipe(
            args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images[0]

        image.save(filepath)
        print(f"[INFO] Seed {seed} saved to {filepath}")

    print("[INFO] Batch generation complete!")


if __name__ == "__main__":
    main()
