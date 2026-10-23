#!/usr/bin/env python3
"""
Deploy ClassScript Gradio App to Hugging Face Spaces (ZeroGPU).
Creates a pristine staging folder with only the essential code and manifests,
avoiding local virtualenvs, model weights, cache, and large archives.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi, get_token


def main():
    repo_id = os.environ.get("HF_SPACE_REPO_ID", "siddiqrossli/classscript-app")
    token = get_token() or os.environ.get("HF_TOKEN")

    if not token:
        print("[ERROR] No Hugging Face token found. Run 'huggingface-cli login' first.", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[1]
    staging_dir = project_root / ".deploy_staging"

    print(f"[1/3] Preparing clean staging directory at: {staging_dir}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # 1. Copy top-level manifests
    top_files = ["README.md", "app.py", "packages.txt", "requirements.txt"]
    for fname in top_files:
        src_file = project_root / fname
        if src_file.is_file():
            shutil.copy2(src_file, staging_dir / fname)
            print(f"  + Copied {fname}")
        else:
            print(f"  ! Warning: {fname} not found!")

    # 2. Copy src directory (excluding pycache, uploads, transcripts, images, env)
    def ignore_filter(dir_path, names):
        ignored = set()
        for name in names:
            if name in {"__pycache__", "uploads", "transcripts", "images", ".git"}:
                ignored.add(name)
            elif name.endswith((".pyc", ".env", ".log", ".wav", ".mp3", ".rar", ".zip")):
                ignored.add(name)
        return ignored

    src_dest = staging_dir / "src"
    shutil.copytree(project_root / "src", src_dest, ignore=ignore_filter)
    print("  + Copied clean src/ tree")

    # Create empty .gitkeep inside uploads and data/transcripts so folders exist
    (src_dest / "web" / "uploads").mkdir(parents=True, exist_ok=True)
    (src_dest / "web" / "uploads" / ".gitkeep").touch()
    (src_dest / "web" / "data" / "transcripts").mkdir(parents=True, exist_ok=True)
    (src_dest / "web" / "data" / "transcripts" / ".gitkeep").touch()

    # Verify staged files
    staged_files = list(staging_dir.rglob("*"))
    print(f"\n[2/3] Staging complete: {len([f for f in staged_files if f.is_file()])} files prepared.")

    # 3. Upload to Hugging Face Space (prunes remote files like unwanted .venv)
    print(f"\n[3/3] Uploading and syncing to Hugging Face Space: {repo_id}...")
    api = HfApi()
    commit_info = api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="space",
        token=token,
        delete_patterns=["*"],
        commit_message="Clean sync ClassScript Gradio portfolio app to Hugging Face ZeroGPU",
    )

    # Cleanup staging directory
    try:
        shutil.rmtree(staging_dir)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("[SUCCESS] Deployment upload & sync successful!")
    print(f"Commit URL: {commit_info}")
    print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
    print(f"Direct App: https://{repo_id.replace('/', '-')}.hf.space")
    print("=" * 60)



if __name__ == "__main__":
    main()
