# src/pipeline/config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    base_dir: str = "D:/final year project/datasets/ted_extracted/"
    output_dir: str = "D:/final year project/datasets/ted_extracted/models/"
    model_name: str = "facebook/wav2vec2-base-960h"
    target_sr: int = 16000
    segment_length: int = 15
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 20
    warmup_steps: int = 100
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)