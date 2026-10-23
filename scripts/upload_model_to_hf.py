#!/usr/bin/env python3
"""
Upload fine-tuned ASR model checkpoint to Hugging Face Model Hub.
Allows the Hugging Face Space (or any cloud instance) to automatically
download and cache the model weights during deployment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Upload fine-tuned Wav2Vec2 ASR model to Hugging Face Model Hub."
    )
    default_model_dir = (
        r"D:\final year project\model_100k_clean_v2_1k_lm"
        if os.path.exists(r"D:\final year project\model_100k_clean_v2_1k_lm")
        else r"E:\final year project\model_100k_clean_v2_1k_lm"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model_dir,
        help="Path to local model folder (contains pytorch_model.bin or .safetensors, etc.)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face Model Repo ID (e.g. your-username/asr-lecture-transcriber-model)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private (recommended if you want to keep weights private)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face write token (optional if already logged in via huggingface-cli login)",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_path).resolve()
    if not model_dir.is_dir():
        print(f"[ERROR] Model directory not found: {model_dir}", file=sys.stderr)
        print("Please check --model_path argument.", file=sys.stderr)
        sys.exit(1)

    print(f"Checking model directory: {model_dir}")
    files = list(model_dir.iterdir())
    print(f"Found {len(files)} files/folders in model directory:")
    for f in files[:10]:
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)" if f.is_file() else f"  - [DIR] {f.name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more.")

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("[ERROR] huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    token = args.token
    if token:
        login(token=token)
    else:
        print("\nNote: If not logged in, please provide --token or run 'huggingface-cli login' first.\n")

    api = HfApi()

    print(f"\n[1/2] Creating or verifying model repository: {args.repo_id} (private={args.private})...")
    try:
        repo_url = api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"Repository ready: {repo_url}")
    except Exception as e:
        print(f"[ERROR] Failed to create or access repo {args.repo_id}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[2/2] Uploading model files to Hugging Face Model Hub (this may take a few minutes)...")
    try:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("\n=======================================================")
        print(f"SUCCESS! Model uploaded to: https://huggingface.co/{args.repo_id}")
        print("=======================================================")
        print("Next steps for your Hugging Face Space:")
        print(f"1. Go to your Space on Hugging Face -> Settings -> Variables and secrets")
        print(f"2. Add Secret or Variable:")
        print(f"     Name:  HF_MODEL_REPO_ID")
        print(f"     Value: {args.repo_id}")
        if args.private:
            print(f"3. Add Secret for private repo access:")
            print(f"     Name:  HF_TOKEN")
            print(f"     Value: <your-hf-read-token>")
        print("=======================================================\n")
    except Exception as e:
        print(f"[ERROR] Failed to upload files: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
