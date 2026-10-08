from __future__ import annotations

import os
from pathlib import Path


class WebConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "change-me-in-production"

    # Flask server
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = False

    # Upload settings
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    UPLOAD_FOLDER = str((Path(__file__).resolve().parent / "uploads").resolve())

    ALLOWED_EXTENSIONS = {
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
        ".mp4", ".mov", ".avi", ".mkv", ".webm"
    }

    # Model path
    _DEFAULT_MODEL_PATH = (
        r"D:\final year project\model_100k_clean_v2_1k_lm"
        if os.path.exists(r"D:\final year project\model_100k_clean_v2_1k_lm")
        else r"E:\final year project\model_100k_clean_v2_1k_lm"
    )
    MODEL_PATH = os.environ.get("ASR_MODEL_PATH") or _DEFAULT_MODEL_PATH

    # Audio settings
    TARGET_SR = 16000

    # Post-processing
    USE_POSTPROCESS = True

    # Transcription chunking (long audio)
    CHUNK_SECONDS = 30

    # Transcript analysis
    KEYWORD_COUNT = 12
    SUMMARY_SENTENCES = 3
    READING_WPM = 200

    # Delete uploads older than this many hours (0 = disabled)
    UPLOAD_MAX_AGE_HOURS = 24

    # Saved transcripts for library / RAG (not used by ASR model)
    TRANSCRIPT_LIBRARY_DIR = str(
        (Path(__file__).resolve().parent / "data" / "transcripts").resolve()
    )

    # Optional LLM features (see ai_service.py and .env.example)
    # ASR model path and pipeline are unchanged when AI is enabled.

    @classmethod
    def init_app(cls) -> None:
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.TRANSCRIPT_LIBRARY_DIR, exist_ok=True)