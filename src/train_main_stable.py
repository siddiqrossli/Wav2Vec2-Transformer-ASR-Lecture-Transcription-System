#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import soundfile as sf
import torch
from torch.utils.data import Dataset

import transformers
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import Trainer, TrainingArguments, TrainerCallback

from pipeline.model import ASRModel
from pipeline.preprocessor import AudioPreprocessor
from pipeline.config import Config
from normalize_text import normalize_text


BASE_DIR_DEFAULT = "D:/final year project/datasets/ted_extracted/"
BASE_MODEL_DEFAULT = r"D:\final year project\datasets\ted_extracted\models\model_100k"


def pick_csv(base_dir: str, name: str) -> str:
    """Prefer *_clean.csv when present."""
    base = Path(base_dir)
    clean = base / f"{name}_clean.csv"
    normal = base / f"{name}.csv"
    if clean.exists():
        return str(clean)
    return str(normal)


def wav_duration_s(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    out = Path(output_dir)
    if not out.exists():
        return None

    checkpoints = []
    for p in out.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-")[-1])
                checkpoints.append((step, str(p)))
            except ValueError:
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def copy_processor_to_checkpoint(base_model_path: str, checkpoint_path: str):
    """Copy processor files from base model to checkpoint folder."""
    files_to_copy = [
        "preprocessor_config.json",
        "tokenizer_config.json", 
        "vocab.json",
        "special_tokens_map.json",
        "added_tokens.json"
    ]
    
    copied = []
    for fname in files_to_copy:
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(checkpoint_path, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied.append(fname)
    
    if copied:
        print(f"  Copied processor files to checkpoint: {', '.join(copied)}")


class NaNGuard(TrainerCallback):
    """Stops training if NaNs/Infs appear in model parameters."""
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        with torch.no_grad():
            for p in model.parameters():
                if p is None:
                    continue
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print("\n[ERROR] NaN/Inf detected in parameters. Stopping training.")
                    control.should_training_stop = True
                    break
        return control


class TedSegmentCSVDataset(Dataset):
    """
    Loads already-cut segment wavs + their transcript using standard library.
    NO PANDAS.
    """
    def __init__(
        self,
        csv_path: str,
        model_wrapper: ASRModel,
        preprocessor: AudioPreprocessor,
        max_samples: Optional[int] = None,
        min_dur_s: float = 0.30,
    ):
        self.csv_path = csv_path
        self.model_wrapper = model_wrapper
        self.preprocessor = preprocessor
        self.min_dur_s = min_dur_s

        all_rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

        if max_samples and len(all_rows) > max_samples:
            import random
            random.seed(42)
            all_rows = random.sample(all_rows, max_samples)

        keep_rows = []
        skipped = 0
        for r in all_rows:
            wav_path = str(r.get("wav_path", "")).strip()
            if not os.path.exists(wav_path):
                skipped += 1
                continue

            try:
                if wav_duration_s(wav_path) < self.min_dur_s:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            text_norm = normalize_text(r.get("text", ""))
            if not text_norm:
                skipped += 1
                continue

            keep_rows.append((wav_path, text_norm))

        self.items = keep_rows
        print(f"[Dataset] Loaded {len(self.items)} usable rows from {csv_path} (skipped {skipped})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wav_path, text_norm = self.items[idx]

        processed = self.preprocessor.process(wav_path)
        audio = processed["full_audio"]

        inputs = self.model_wrapper.get_input_features(audio)
        labels = self.model_wrapper.processor.tokenizer(text_norm).input_ids

        return {
            "input_values": inputs.input_values[0],
            "labels": labels,
        }


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=BASE_DIR_DEFAULT)
    ap.add_argument("--out_dir", default=None, help="Output model dir")
    ap.add_argument("--base_model", default=BASE_MODEL_DEFAULT, help="Starting model path")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--max_val", type=int, default=2000)
    ap.add_argument("--min_dur", type=float, default=0.30)
    ap.add_argument("--fp16", action="store_true", help="Enable fp16")
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in out_dir if available")
    args = ap.parse_args()

    base_dir = args.base_dir
    train_csv = pick_csv(base_dir, "train")
    val_csv = pick_csv(base_dir, "val")

    out_dir = args.out_dir or os.path.join(base_dir, "models", "model_stable")
    latest_checkpoint = find_latest_checkpoint(out_dir) if args.resume else None

    print("=" * 70)
    print("STABLE TRAINING (NO PANDAS)")
    print("=" * 70)
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV  : {val_csv}")
    print(f"Out dir  : {out_dir}")
    print(f"Base model: {args.base_model}")
    print(f"epochs={args.epochs} batch={args.batch} lr={args.lr} max_train={args.max_train} max_val={args.max_val}")
    print(f"min_dur={args.min_dur} fp16={args.fp16} resume={args.resume}")
    if latest_checkpoint:
        print(f"Latest checkpoint found: {latest_checkpoint}")
        # FIX: Copy processor files to checkpoint so it can load
        copy_processor_to_checkpoint(args.base_model, latest_checkpoint)
    print("=" * 70)

    cfg = Config(
    base_dir=base_dir,
    output_dir=os.path.join(base_dir, "models"),
    num_epochs=args.epochs,
    batch_size=args.batch,
    learning_rate=args.lr,
    max_train_samples=args.max_train,
    max_val_samples=args.max_val,
    )

    cfg.augment = True

    preprocessor = AudioPreprocessor(cfg)
    model_wrapper = ASRModel(cfg)

    if latest_checkpoint:
        print(f"Resuming model weights from checkpoint: {latest_checkpoint}")
        model_wrapper.load(latest_checkpoint)
    else:
        print(f"Loading model from: {args.base_model}")
        model_wrapper.load(args.base_model)

    model_wrapper.model.config.pad_token_id = model_wrapper.processor.tokenizer.pad_token_id
    model_wrapper.model.config.ctc_zero_infinity = True

    train_ds = TedSegmentCSVDataset(
        csv_path=train_csv,
        model_wrapper=model_wrapper,
        preprocessor=preprocessor,
        max_samples=args.max_train,
        min_dur_s=args.min_dur,
    )
    val_ds = TedSegmentCSVDataset(
        csv_path=val_csv,
        model_wrapper=model_wrapper,
        preprocessor=preprocessor,
        max_samples=args.max_val,
        min_dur_s=args.min_dur,
    )

    model_wrapper.freeze_feature_encoder()

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())

    train_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=200,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=use_fp16,
        report_to=[],
        remove_unused_columns=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model_wrapper.model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorCTCWithPadding(model_wrapper.processor),
        callbacks=[NaNGuard()],
    )

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    print("\nSaving final model...")
    model_wrapper.save(out_dir)

    print("\nDONE.")
    print(f"Model saved at: {out_dir}")


if __name__ == "__main__":
    main()