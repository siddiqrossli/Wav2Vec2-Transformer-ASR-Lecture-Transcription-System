# src/pipeline/model.py
from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    Wav2Vec2ForCTC,
)

from .config import Config


class ASRModel:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.use_lm = False

    def freeze_feature_encoder(self) -> None:
        if self.model is not None:
            self.model.freeze_feature_encoder()

    def get_input_features(self, audio: np.ndarray):
        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.processor(
            audio,
            sampling_rate=self.config.target_sr,
            return_tensors="pt",
            padding=True,
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        input_values = input_values.to(self.device)
        outputs = self.model(input_values)
        return outputs.logits

    def decode(self, logits: torch.Tensor, decode_mode: str = "greedy") -> str:
        """
        Decode model logits.

        decode_mode="greedy":
            Greedy CTC decoding.

        decode_mode="lm":
            N-gram language model beam-search decoding.
            Requires Wav2Vec2ProcessorWithLM.
        """
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Call load() first.")

        decode_mode = (decode_mode or "greedy").lower().strip()

        if decode_mode == "lm":
            if not isinstance(self.processor, Wav2Vec2ProcessorWithLM):
                raise RuntimeError(
                )
            decoded = self.processor.batch_decode(
                logits.detach().cpu().numpy()
            )
            text = decoded.text[0]

        else:
            pred_ids = torch.argmax(logits, dim=-1)
            pred_ids_np = pred_ids.detach().cpu().numpy()
            if isinstance(self.processor, Wav2Vec2ProcessorWithLM):
                text = self.processor.tokenizer.batch_decode(pred_ids_np)[0]
            else:
                text = self.processor.batch_decode(pred_ids_np)[0]
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """
        Basic cleaning only.
        Rule-based correction is handled separately in postprocess_corrections.py.
        """
        if not text:
            return ""

        t = str(text).upper()

        for tok in ["<PAD>", "<S>", "</S>", "[CLS]", "[SEP]", "<START>", "<END>"]:
            t = t.replace(tok, "")

        t = t.replace("|", " ")
        t = re.sub(r"\s+", " ", t).strip()

        return t

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, decode_mode: str = "greedy") -> str:
        """
        Transcribe audio using either greedy CTC decoding or LM decoding.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.get_input_features(audio)
        logits = self.forward(inputs.input_values)

        return self.decode(logits, decode_mode=decode_mode)

    def save(self, path: str) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("No model to save.")

        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

        print(f"Model saved to {path}")

    def load(
        self,
        path: str,
        processor_fallback_path: Optional[str] = None,
        use_lm: bool = False,
    ) -> None:
        """
        Load Wav2Vec2 model and processor.

        use_lm=False:
            Loads normal Wav2Vec2Processor for greedy decoding.

        use_lm=True:
            Loads Wav2Vec2ProcessorWithLM for n-gram LM beam-search decoding.
            This requires language model files inside the model folder.
        """
        print(f"Loading model from: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path not found: {path}")

        contents = os.listdir(path)

        has_pytorch = any(f.startswith("pytorch_model") for f in contents)
        has_safetensors = any(f.endswith(".safetensors") for f in contents)

        if not has_pytorch and not has_safetensors:
            raise FileNotFoundError(f"No model weights found in {path}!")

        try:
            if use_lm:
                self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(path)
                self.use_lm = True
                print(f"  Loaded processor WITH language model from: {path}")
            else:
                self.processor = Wav2Vec2Processor.from_pretrained(path)
                self.use_lm = False
                print(f"  Loaded processor from: {path}")

        except OSError as e:
            if processor_fallback_path and os.path.exists(processor_fallback_path):
                print(
                    f"  Processor not found in checkpoint, loading from: "
                    f"{processor_fallback_path}"
                )

                if use_lm:
                    self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(
                        processor_fallback_path
                    )
                    self.use_lm = True
                    print(
                        f"  Loaded fallback processor WITH language model from: "
                        f"{processor_fallback_path}"
                    )
                else:
                    self.processor = Wav2Vec2Processor.from_pretrained(
                        processor_fallback_path
                    )
                    self.use_lm = False
                    print(f"  Loaded fallback processor from: {processor_fallback_path}")

            else:
                raise e

        self.model = Wav2Vec2ForCTC.from_pretrained(path).to(self.device)
        self.model.eval()

        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.ctc_zero_infinity = True

        print(f"Successfully loaded model from: {path}")
        print(f"Device: {self.device}")
        print(f"LM decoding enabled: {self.use_lm}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")