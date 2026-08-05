# src/pipeline/inference.py
from __future__ import annotations

from typing import Union, List, Dict
import numpy as np


class TranscriptionPipeline:
    def __init__(self, model_wrapper, preprocessor):
        self.model = model_wrapper
        self.preprocessor = preprocessor

    def transcribe_audio(
        self,
        audio_input: Union[str, np.ndarray],
        original_sr: int | None = None,
    ) -> Dict:
        # Preprocess audio
        processed = self.preprocessor.process(audio_input, original_sr)

        # Transcribe each segment
        segments_text = []
        for i, segment in enumerate(processed["segments"], 1):
            # Use greedy decoding (fastest, no parameters needed)
            text = self.model.transcribe(segment)
            segments_text.append(text)
            print(f"  Segment {i}/{len(processed['segments'])}: {text[:60]}...")

        # Combine all segments
        full_text = " ".join(t for t in segments_text if t).strip()
        
        return {
            "text": full_text,
            "segments": segments_text,
            "num_segments": len(segments_text),
            "duration": len(processed["full_audio"]) / self.preprocessor.config.target_sr,
        }

    def transcribe_batch(self, audio_paths: List[str]) -> List[Dict]:
        results = []
        for i, path in enumerate(audio_paths, 1):
            print(f"\nProcessing {i}/{len(audio_paths)}: {path}")
            result = self.transcribe_audio(path)
            results.append({"path": path, **result})
        return results