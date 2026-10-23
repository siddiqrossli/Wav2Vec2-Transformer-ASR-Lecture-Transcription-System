from __future__ import annotations

import os
from pathlib import Path


def resolve_model_path() -> str:
    """
    Resolve model path from local disk, environment variable, or download
    from Hugging Face Model Hub if missing (for cloud / Hugging Face Spaces).
    """
    explicit_path = os.environ.get("ASR_MODEL_PATH")
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    # Check common local candidates
    local_candidates = [
        explicit_path,
        str((Path(__file__).resolve().parents[2] / "model_100k_clean_v2_1k_lm").resolve()),
        str((Path(__file__).resolve().parent / "model_weights").resolve()),
        r"D:\final year project\model_100k_clean_v2_1k_lm",
        r"E:\final year project\model_100k_clean_v2_1k_lm",
    ]
    for candidate in local_candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    # If model is not found locally, check if Hugging Face repo ID is configured
    hf_repo = os.environ.get("HF_MODEL_REPO_ID")
    if hf_repo:
        try:
            from huggingface_hub import snapshot_download

            hf_token = os.environ.get("HF_TOKEN")
            cache_dir = os.environ.get("HF_CACHE_DIR") or str(
                (Path(__file__).resolve().parent / "model_weights").resolve()
            )
            print(f"[HF_HUB] Model not found locally. Downloading from Hugging Face Hub: {hf_repo}...")
            downloaded_dir = snapshot_download(
                repo_id=hf_repo,
                token=hf_token,
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
            print(f"[HF_HUB] Model downloaded successfully to: {downloaded_dir}")
            return downloaded_dir
        except Exception as e:
            print(f"[HF_HUB] Error downloading model from Hugging Face Hub: {e}")

    # Fallback to default path string if nothing matched
    return explicit_path or r"D:\final year project\model_100k_clean_v2_1k_lm"


class WebConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "change-me-in-production"

    # Flask server settings
    # HF Spaces automatically sets SPACE_ID and exposes port 7860
    PORT = int(os.environ.get("PORT", 7860 if os.environ.get("SPACE_ID") else 5000))
    HOST = os.environ.get("HOST", "0.0.0.0")
    DEBUG = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}

    # Upload settings
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "500")) * 1024 * 1024  # 500MB default
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER") or str((Path(__file__).resolve().parent / "uploads").resolve())

    ALLOWED_EXTENSIONS = {
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
        ".mp4", ".mov", ".avi", ".mkv", ".webm"
    }

    # Model path
    MODEL_PATH = resolve_model_path()

    # Audio settings
    TARGET_SR = 16000

    # Post-processing
    USE_POSTPROCESS = True

    # Transcription chunking (long audio)
    CHUNK_SECONDS = int(os.environ.get("CHUNK_SECONDS", "30"))

    # Max audio duration for online processing (protects against ZeroGPU 120s timeout)
    # Defaults to 35 minutes when running in Hugging Face Space (SPACE_ID present), 0 (unlimited) locally
    MAX_AUDIO_MINUTES = int(os.environ.get("MAX_AUDIO_MINUTES", "35" if os.environ.get("SPACE_ID") else "0"))

    # Transcript analysis
    KEYWORD_COUNT = 12
    SUMMARY_SENTENCES = 3
    READING_WPM = 200

    # Delete uploads older than this many hours (0 = disabled)
    UPLOAD_MAX_AGE_HOURS = int(os.environ.get("UPLOAD_MAX_AGE_HOURS", "24"))

    # Saved transcripts for library / RAG (not used by ASR model)
    TRANSCRIPT_LIBRARY_DIR = os.environ.get("TRANSCRIPT_LIBRARY_DIR") or str(
        (Path(__file__).resolve().parent / "data" / "transcripts").resolve()
    )

    # Optional LLM features (see ai_service.py and .env.example)
    # ASR model path and pipeline are unchanged when AI is enabled.

    @classmethod
    def init_app(cls) -> None:
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.TRANSCRIPT_LIBRARY_DIR, exist_ok=True)