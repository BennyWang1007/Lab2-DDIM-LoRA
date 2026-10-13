import json
import shutil
import subprocess
import zipfile
from typing import Union

from datasets import load_dataset
from pathlib import Path


def setup_senko_san_dataset():
    """
    Downloads the Senko-san dataset from Hugging Face and saves a curated subset of images for DreamBooth training.
    """
    # Load dataset from Hugging Face
    ds = load_dataset("JustInEchoes/Senko-san-NAI-V5-Curated-HD-ULTRA-Tags", split="train")

    save_dir = Path("./data/senko_san")
    save_dir.mkdir(parents=True, exist_ok=True)

    # DreamBooth works best with 5 to 30 curated images
    for idx, item in enumerate(ds):
        if idx >= 30:  # Select a small, diverse subset for DreamBooth
            break
        item["image"].convert("RGB").save(save_dir / f"senko_{idx:03d}.png")


def setup_gambit_dataset(
    repo_url: str = "https://huggingface.co/datasets/EarthnDusk/Gambit_Dataset_and_Lora",
    data_root: Union[str, Path] = "./data",
) -> Path:
    """Clones the Gambit dataset repo, unzips the training images/captions,

    cleans up macOS artifacts and temporary folders, and builds metadata.jsonl.

    Args:
        repo_url: Hugging Face dataset git URL.
        data_root: Directory to store the raw repo and extracted dataset.

    Returns:
        Path to the prepared dataset directory.
    """
    data_root = Path(data_root)
    repo_dir = data_root / "Gambit_Dataset_and_Lora"
    target_dir = data_root / "RemyLebeau_dataset"
    zip_path = repo_dir / "RemyLebeau_dataset.zip"
    temp_extract_dir = data_root / "_temp_extract"

    data_root.mkdir(parents=True, exist_ok=True)

    # 1. Clone the repository if not already cloned
    if not repo_dir.exists():
        print(f"Cloning {repo_url} into {repo_dir}...")
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    else:
        print(f"Repository directory already exists: {repo_dir}")

    if not zip_path.exists():
        raise FileNotFoundError(f"Expected archive not found: {zip_path}")

    # 2. Extract the archive into a temporary folder
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_extract_dir)

    # 3. Locate source directory and move files to target directory
    source_dir = (
        temp_extract_dir
        / "content"
        / "LoRA"
        / "RemyLebeau_dataset"
        / "train_data"
        / "train_data"
        / "10_sks RemyLebeau"
    )

    if not source_dir.exists():
        # Fallback search if path hierarchy varies
        candidates = list(temp_extract_dir.rglob("10_sks RemyLebeau"))
        if candidates:
            source_dir = candidates[0]
        else:
            raise FileNotFoundError(
                f"Could not locate image directory inside {temp_extract_dir}"
            )

    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        # Filter out macOS resource-fork artifacts (._*)
        if item.name.startswith("._"):
            continue
        dest_item = target_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest_item)

    # Clean up temporary extracted folder
    shutil.rmtree(temp_extract_dir, ignore_errors=True)

    # 4. Generate Hugging Face / Diffusers-compatible metadata.jsonl
    metadata_path = target_dir / "metadata.jsonl"
    valid_exts = {".jpg", ".jpeg", ".png"}

    image_files = sorted(
        [
            p
            for p in target_dir.iterdir()
            if p.suffix.lower() in valid_exts and not p.name.startswith("._")
        ]
    )

    with open(metadata_path, "w", encoding="utf-8") as f_out:
        for img_path in image_files:
            txt_path = target_dir / f"{img_path.stem}.txt"
            caption = ""
            if txt_path.exists():
                caption = txt_path.read_text(encoding="utf-8").strip()

            line = json.dumps(
                {"file_name": img_path.name, "text": caption},
                ensure_ascii=False,
            )
            f_out.write(line + "\n")

    print(
        f"Prepared {len(image_files)} samples with metadata: {metadata_path}"
    )
    return target_dir


if __name__ == "__main__":
    setup_senko_san_dataset()
    setup_gambit_dataset()
