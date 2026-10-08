from __future__ import annotations

import os
import glob
import csv
from typing import List, Dict, Optional

from .config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def load_ted_data(self, split: str = "train") -> List[Dict]:
        """
        Load TED segment dataset from CSV using standard library only.

        Supported splits:
          - train -> train.csv
          - val / validation -> val.csv
          - test -> test.csv
        """
        split_map = {
            "train": "train.csv",
            "val": "val.csv",
            "validation": "val.csv",
            "test": "test.csv",
        }

        if split not in split_map:
            raise ValueError(f"Unknown split='{split}'. Use train/val/test.")

        csv_path = os.path.join(self.config.base_dir, split_map[split])
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Use standard library csv instead of pandas
        data: List[Dict] = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "audio_path": row["wav_path"],
                    "text": row["text"],
                    "source": "ted",
                })

        # Limit samples if specified
        if self.config.max_train_samples and split == "train":
            data = data[:self.config.max_train_samples]
        if self.config.max_val_samples and split in {"val", "validation"}:
            data = data[:self.config.max_val_samples]

        print(f"Loaded {len(data)} samples from {csv_path}")
        return data

    def load_local_librispeech(self, root_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load local LibriSpeech data from disk.
        """
        print(f"Loading local LibriSpeech from {root_dir}...")

        transcript_files = glob.glob(os.path.join(root_dir, "**/*.trans.txt"), recursive=True)
        print(f"Found {len(transcript_files)} transcript files")

        data: List[Dict] = []
        for trans_file in transcript_files:
            dir_path = os.path.dirname(trans_file)
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue

                    utt_id, text = parts
                    audio_file = os.path.join(dir_path, f"{utt_id}.flac")

                    if os.path.exists(audio_file):
                        data.append({
                            "audio_path": audio_file,
                            "text": text.upper(),
                            "source": "librispeech_local",
                        })

                        if max_samples and len(data) >= max_samples:
                            print(f"Loaded {len(data)} samples from local LibriSpeech")
                            return data

        print(f"Loaded {len(data)} samples from local LibriSpeech")
        return data

    def load_custom_audio(self, audio_path: str) -> Dict:
        """
        Load one audio path for inference.
        """
        return {
            "audio_path": audio_path,
            "text": None,
            "source": "custom",
        }

    def combine_datasets(self, *datasets: List[Dict]) -> List[Dict]:
        """
        Combine multiple datasets.
        """
        combined: List[Dict] = []
        for ds in datasets:
            combined.extend(ds)
        return combined