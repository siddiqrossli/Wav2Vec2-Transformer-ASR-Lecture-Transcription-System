from __future__ import annotations

import os
import sys
import wave
from pathlib import Path
from typing import Dict, Union, Tuple, List

import numpy as np

# Allow importing normalize_text from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalize_text import normalize_text  # noqa: E402

from .config import Config


def augment_audio(audio: np.ndarray) -> np.ndarray:

    if audio.size == 0:
        return audio.astype(np.float32)

    # 1. Add small noise
    noise = np.random.normal(0, 0.003, audio.shape)
    audio = audio + noise

    # 2. Random volume scaling
    volume_scale = np.random.uniform(0.9, 1.1)
    audio = audio * volume_scale

    # 3. Small time shift
    shift = np.random.randint(-200, 200)
    audio = np.roll(audio, shift)

    # Keep audio inside [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)


class AudioPreprocessor:
    def __init__(self, config: Config):
        self.config = config

    def _is_precut_segment_path(self, audio_input: Union[str, np.ndarray]) -> bool:
        """
        Heuristic to detect already-cut segments.
        """
        if not isinstance(audio_input, str):
            return False

        p = Path(audio_input)
        name = p.name.lower()
        parent = p.parent.name.lower()

        return ("_seg" in name) or (parent in {"segments_wav", "segments"})

    def read_wav(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Read WAV file using ONLY Python standard library wave module.
        NO external dependencies.
        """
        with wave.open(filepath, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            raw_data = wav_file.readframes(n_frames)

            if sample_width == 1:
                data = np.frombuffer(raw_data, dtype=np.uint8)
                data = (data.astype(np.float32) - 128.0) / 128.0

            elif sample_width == 2:
                data = np.frombuffer(raw_data, dtype=np.int16)
                data = data.astype(np.float32) / 32768.0

            elif sample_width == 4:
                data = np.frombuffer(raw_data, dtype=np.int32)
                data = data.astype(np.float32) / 2147483648.0

            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)

            return data.astype(np.float32), sample_rate

    def write_wav(self, filepath: str, audio: np.ndarray, sample_rate: int = 16000):
        """
        Write WAV file using ONLY Python standard library wave module.
        """
        audio = np.clip(audio, -1.0, 1.0)
        audio_int = (audio * 32767).astype(np.int16)

        with wave.open(filepath, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())

    def resample_linear(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Linear interpolation resampling.
        NO torchaudio, NO external libraries.
        """
        if orig_sr == target_sr:
            return audio.astype(np.float32)

        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)

        if new_length <= 0:
            return audio.astype(np.float32)

        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)

        resampled = np.interp(new_indices, old_indices, audio)

        return resampled.astype(np.float32)

    def energy_based_vad(
        self,
        audio: np.ndarray,
        frame_length_ms: int = 30,
        energy_threshold: float = 0.005,
        min_speech_duration_ms: int = 250,
    ) -> List[Tuple[int, int]]:
        """
        Energy-based Voice Activity Detection.
        NO Silero VAD, NO external dependencies.
        """
        sample_rate = self.config.target_sr
        frame_length = int(sample_rate * frame_length_ms / 1000)
        hop_length = frame_length // 2

        if len(audio) < frame_length:
            return [(0, len(audio))]

        num_frames = (len(audio) - frame_length) // hop_length + 1
        energies = []

        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]

            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)

        energies = np.array(energies)

        if energies.max() > 0:
            threshold = max(energy_threshold, energies.max() * 0.05)
        else:
            return [(0, len(audio))]

        is_speech = energies > threshold

        segments = []
        min_speech_frames = int(min_speech_duration_ms / (hop_length / sample_rate * 1000))

        in_speech = False
        start_frame = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_frame = i
                in_speech = True

            elif not speech and in_speech:
                if i - start_frame >= min_speech_frames:
                    start_sample = start_frame * hop_length
                    end_sample = min(i * hop_length + frame_length, len(audio))
                    segments.append((start_sample, end_sample))

                in_speech = False

        if in_speech and len(is_speech) - start_frame >= min_speech_frames:
            start_sample = start_frame * hop_length
            end_sample = len(audio)
            segments.append((start_sample, end_sample))

        if not segments:
            return [(0, len(audio))]

        return segments

    def merge_close_segments(
        self,
        segments: List[Tuple[int, int]],
        max_gap_ms: float = 500,
    ) -> List[Tuple[int, int]]:
        """
        Merge segments that are close together.
        """
        if not segments:
            return []

        max_gap_samples = int(max_gap_ms * self.config.target_sr / 1000)
        segments = sorted(segments)

        merged = [list(segments[0])]

        for start, end in segments[1:]:
            last_start, last_end = merged[-1]

            if start - last_end <= max_gap_samples:
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])

        return [(s, e) for s, e in merged]

    def split_long_segments(
        self,
        segments: List[Tuple[int, int]],
        max_duration_ms: float = 15000,
    ) -> List[Tuple[int, int]]:
        """
        Split segments that are too long.
        """
        max_samples = int(max_duration_ms * self.config.target_sr / 1000)
        result = []

        for start, end in segments:
            duration = end - start

            if duration <= max_samples:
                result.append((start, end))
            else:
                current = start
                while current < end:
                    next_end = min(current + max_samples, end)
                    result.append((current, next_end))
                    current = next_end

        return result

    def remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove silence using hardcoded energy-based VAD.
        """
        segments = self.energy_based_vad(audio)
        segments = self.merge_close_segments(segments)
        segments = self.split_long_segments(segments)

        speech_parts = []

        for start, end in segments:
            chunk = audio[start:end]
            if len(chunk) > 0:
                speech_parts.append(chunk)

        if speech_parts:
            return np.concatenate(speech_parts).astype(np.float32)

        return audio.astype(np.float32)

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize into [-1, 1] only if needed.
        """
        if audio.size == 0:
            return audio.astype(np.float32)

        mx = float(np.max(np.abs(audio)))

        if mx > 1.0:
            audio = audio / mx

        return audio.astype(np.float32)

    def segment(self, audio: np.ndarray) -> list[np.ndarray]:
        """
        Fixed-length segmentation.
        """
        seg_samples = int(self.config.segment_length * self.config.target_sr)

        if audio.size == 0:
            return [audio.astype(np.float32)]

        if len(audio) <= seg_samples:
            return [audio.astype(np.float32)]

        segments = []

        for i in range(0, len(audio), seg_samples):
            seg = audio[i:i + seg_samples]

            if len(seg) < seg_samples:
                seg = np.pad(seg, (0, seg_samples - len(seg)), mode="constant")

            segments.append(seg.astype(np.float32))

        return segments

    def load_audio(
        self,
        audio_input: Union[str, np.ndarray],
        original_sr: int = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from file or array.
        """
        if isinstance(audio_input, str):
            audio, sr = self.read_wav(audio_input)
        else:
            audio = audio_input.astype(np.float32)
            sr = int(original_sr) if original_sr is not None else self.config.target_sr

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio.astype(np.float32), sr

    def process(
        self,
        audio_input: Union[str, np.ndarray],
        original_sr: int = None,
    ) -> Dict:
        """
        Full preprocessing.
        """
        audio, sr = self.load_audio(audio_input, original_sr)
        audio = self.resample_linear(audio, sr, self.config.target_sr)
        audio = self.normalize(audio)

        if getattr(self.config, "augment", False):
            audio = augment_audio(audio)

        if self._is_precut_segment_path(audio_input):
            return {
                "segments": [audio],
                "full_audio": audio,
                "sample_rate": self.config.target_sr,
            }

        #audio = self.remove_silence(audio)
        segments = self.segment(audio)

        return {
            "segments": segments,
            "full_audio": audio,
            "sample_rate": self.config.target_sr,
        }

    def process_text(self, text: str) -> str:
        """
        Normalize text for Wav2Vec2 CTC training.
        """
        return normalize_text(text)