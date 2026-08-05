# src/pipeline/trainer.py
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
from typing import Dict, List, Optional
import sys
import os
import logging

logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalize_text import normalize_text

# Import augmentation
try:
    from safe_to_delete.augmentation import AudioAugmenter
except ImportError:
    AudioAugmenter = None
    logger.warning("AudioAugmenter not found, augmentation disabled")


@dataclass
class DataCollatorCTCWithPadding:
    processor: any
    padding: bool = True

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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


class ASRDataset(Dataset):
    def __init__(self, data: List[Dict], preprocessor, model_wrapper, 
                 use_random_segment: bool = True,
                 augment: bool = False,
                 augment_prob: float = 0.5):
        self.data = data
        self.preprocessor = preprocessor
        self.model_wrapper = model_wrapper
        self.use_random_segment = use_random_segment
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Initialize augmenter if needed
        self.augmenter = None
        if self.augment and AudioAugmenter is not None:
            self.augmenter = AudioAugmenter(sample_rate=16000)
            print(f"  Augmentation enabled (p={augment_prob})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process audio
        if "audio" in item:
            audio = item["audio"]
            sr = item.get("sampling_rate", 16000)
            processed = self.preprocessor.process(audio, sr)
        else:
            processed = self.preprocessor.process(item["audio_path"])
        
        # Handle segments
        if processed["segments"]:
            if self.use_random_segment and len(processed["segments"]) > 1:
                seg_idx = random.randint(0, len(processed["segments"]) - 1)
                audio_segment = processed["segments"][seg_idx]
            else:
                audio_segment = processed["segments"][0]
        else:
            audio_segment = processed["full_audio"]
        
        # APPLY AUGMENTATION (only during training)
        if self.augment and self.augmenter is not None and random.random() < self.augment_prob:
            audio_segment = self.augmenter.apply_random(audio_segment)
        
        # Get input features
        inputs = self.model_wrapper.get_input_features(audio_segment)
        
        # Process text
        text = self.preprocessor.process_text(item["text"])
        labels = self.model_wrapper.processor.tokenizer(text).input_ids
        
        return {
            "input_values": inputs.input_values[0],
            "labels": labels
        }


class ASRTrainer:
    def __init__(self, model_wrapper, config):
        self.model_wrapper = model_wrapper
        self.config = config
        self.data_collator = DataCollatorCTCWithPadding(model_wrapper.processor)
    
    def prepare_data(self, data: List[Dict], preprocessor, 
                     use_random_segment: bool = True,
                     augment: bool = False,
                     augment_prob: float = 0.5) -> ASRDataset:
        """Prepare dataset for training"""
        return ASRDataset(
            data, 
            preprocessor, 
            self.model_wrapper, 
            use_random_segment=use_random_segment,
            augment=augment,
            augment_prob=augment_prob
        )
    
    def train(self, train_dataset, val_dataset, output_dir: str, 
              resume_from: Optional[str] = None):
        """Run training with resume support"""
        
        self.model_wrapper.freeze_feature_encoder()
        
        total_steps = len(train_dataset) // self.config.batch_size * self.config.num_epochs
        warmup_steps = min(self.config.warmup_steps, total_steps // 10)
        
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            warmup_steps=warmup_steps,
            logging_steps=5,              # Log every 5 steps
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=torch.cuda.is_available(),
            report_to=[],
            remove_unused_columns=False,
            lr_scheduler_type="cosine",
        )
        
        trainer = Trainer(
            model=self.model_wrapper.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.data_collator,
        )
        
        # Determine checkpoint to resume from
        checkpoint = None
        if resume_from and os.path.isdir(resume_from):
            checkpoint = resume_from
            logger.info(f"Resuming from specified checkpoint: {checkpoint}")
        elif os.path.isdir(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) 
                          if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoint = os.path.join(output_dir, sorted(checkpoints)[-1])
                logger.info(f"Auto-resuming from: {checkpoint}")
        
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=checkpoint)
        self.model_wrapper.save(output_dir)
        logger.info(f"Training complete. Model saved to {output_dir}")
        
        return trainer