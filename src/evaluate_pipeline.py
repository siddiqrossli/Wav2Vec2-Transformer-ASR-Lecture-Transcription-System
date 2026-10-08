#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import Config, AudioPreprocessor, ASRModel, Evaluator, TranscriptionPipeline
from pipeline.evaluator import edit_distance
from postprocess_corrections import apply_postprocess, is_garbage


def clean_text(text: str, debug: bool = False, is_reference: bool = False) -> str:
    """
    Clean text before WER/CER evaluation.

    Reference:
    - Uppercase
    - Remove unsupported characters
    - Normalize whitespace
    - Do NOT apply ASR post-processing correction

    Hypothesis:
    - Uppercase
    - Remove unsupported characters
    - Normalize whitespace
    - Apply post-processing correction unless disabled
    """
    original = text if debug else None

    text = str(text or "").upper().strip()

    if "|" in text:
        text = text.replace("|", " ")
        if debug:
            print("  [DEBUG] Converted pipes to spaces")

    elif " " not in text and len(text) > 20:
        if debug:
            print(f"  [WARNING] Text has no spaces or pipes: {text[:50]}...")

    text = re.sub(r"[^A-Z\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not is_reference:
        text = apply_postprocess(text)

    if debug and original is not None:
        print(f"  [DEBUG] Original: {str(original)[:80]}")
        print(f"  [DEBUG] Cleaned : {text[:80]}")

    return text


def calculate_accuracy_metrics(evaluator: Evaluator) -> Dict[str, float]:
    """
    Calculate WER, CER, WRR, word accuracy, character accuracy,
    exact match accuracy, and sentence accuracy.
    """
    metrics = evaluator.get_detailed_report()

    wer = metrics["wer"]
    cer = metrics["cer"]
    wrr = metrics["wrr"]

    word_accuracy = (1.0 - wer) * 100
    character_accuracy = (1.0 - cer) * 100

    total = len(evaluator.all_refs)

    exact_correct = 0
    sentence_accurate = 0

    for ref, hyp in zip(evaluator.all_refs, evaluator.all_hyps):
        ref = ref.strip()
        hyp = hyp.strip()

        if ref == hyp:
            exact_correct += 1

        ref_words = ref.split()
        hyp_words = hyp.split()

        if not ref_words:
            continue

        errors = edit_distance(ref_words, hyp_words)
        sentence_wer = errors / len(ref_words)

        if sentence_wer <= 0.10:
            sentence_accurate += 1

    exact_match_accuracy = (exact_correct / total) * 100 if total > 0 else 0.0
    sentence_accuracy = (sentence_accurate / total) * 100 if total > 0 else 0.0

    return {
        "wer": wer,
        "cer": cer,
        "wrr": wrr,
        "word_accuracy": word_accuracy,
        "character_accuracy": character_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "sentence_accuracy": sentence_accuracy,
        "total_samples": total,
    }


def load_test_data(test_csv: str, num_samples: int | None = None) -> List[Dict[str, str]]:
    """
    Load test CSV with columns:
    - wav_path
    - text
    """
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    rows: List[Dict[str, str]] = []

    with open(test_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if num_samples is not None and i >= num_samples:
                break

            wav_path = str(row.get("wav_path", "")).strip()
            text = str(row.get("text", "")).strip()

            if not wav_path or not text:
                continue

            rows.append({
                "wav_path": wav_path,
                "text": text,
            })

    return rows


def transcribe_with_decode_mode(
    pipeline: TranscriptionPipeline,
    wav_path: str,
    decode_mode: str = "greedy",
) -> str:
    """
    Transcribe one audio file using selected decoding mode.

    decode_mode:
    - greedy: normal CTC greedy decoding
    - lm: n-gram LM beam search decoding
    """
    processed = pipeline.preprocessor.process(wav_path)

    segments_text: List[str] = []

    for i, segment in enumerate(processed["segments"], start=1):
        text = pipeline.model.transcribe(segment, decode_mode=decode_mode)
        text = str(text or "").strip()

        if text:
            segments_text.append(text)

        print(f"  Segment {i}/{len(processed['segments'])}: {text[:60]}...")

    return " ".join(segments_text).strip()


def evaluate(
    pipeline: TranscriptionPipeline,
    test_data: List[Dict[str, str]],
    decode_mode: str = "greedy",
    debug: bool = False,
    max_debug: int = 3,
) -> Evaluator:
    """
    Run ASR evaluation using either:
    - greedy decoding
    - n-gram LM beam search decoding
    """
    evaluator = Evaluator()

    for i, row in enumerate(test_data):
        wav_path = row["wav_path"]

        reference = clean_text(
            row["text"],
            debug=(debug and i < max_debug),
            is_reference=True,
        )

        if debug and i < max_debug:
            print("\n" + "-" * 60)
            print(f"Sample {i}")
            print("-" * 60)
            print(f"Audio: {wav_path}")
            print(f"Raw REF: {str(row['text'])[:120]}")
            print(f"Clean REF: {reference[:120]}")

        try:
            raw_hypothesis = transcribe_with_decode_mode(
                pipeline=pipeline,
                wav_path=wav_path,
                decode_mode=decode_mode,
            )

        except Exception as e:
            print(f"[ERROR] Sample {i}: failed to transcribe")
            print(f"        File: {wav_path}")
            print(f"        Decode mode: {decode_mode}")
            print(f"        Error: {e}")
            raw_hypothesis = ""

        hypothesis = clean_text(
            raw_hypothesis,
            debug=(debug and i < max_debug),
            is_reference=False,
        )

        if is_garbage(hypothesis):
            print(f"[GARBAGE] Sample {i}: output looks invalid, treating as empty.")
            hypothesis = ""

        evaluator.add_sample(reference, hypothesis)

        if debug and i < max_debug:
            print(f"Raw HYP: {str(raw_hypothesis)[:120]}")
            print(f"Clean HYP: {hypothesis[:120]}")

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_data)} samples...")

    return evaluator


def print_results(evaluator: Evaluator, decode_mode: str = "greedy") -> None:
    metrics = calculate_accuracy_metrics(evaluator)

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - {decode_mode.upper()} DECODING")
    print("=" * 70)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Word Error Rate (WER):        {metrics['wer']:.4f} ({metrics['wer'] * 100:.2f}%)")
    print(f"Character Error Rate (CER):   {metrics['cer']:.4f} ({metrics['cer'] * 100:.2f}%)")
    print(f"Word Recognition Rate (WRR):  {metrics['wrr']:.4f} ({metrics['wrr'] * 100:.2f}%)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("ACCURACY SUMMARY")
    print("=" * 70)
    print(f"Word Accuracy (1 - WER):          {metrics['word_accuracy']:.2f}%")
    print(f"Character Accuracy (1 - CER):     {metrics['character_accuracy']:.2f}%")
    print(f"Exact Match Accuracy:             {metrics['exact_match_accuracy']:.2f}%")
    print(f"Sentence Accuracy (<=10% WER):    {metrics['sentence_accuracy']:.2f}%")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("SAMPLE COMPARISONS")
    print("=" * 70)

    comparisons = evaluator.get_sample_comparisons(n=5)

    for comp in comparisons:
        ref = comp["reference"]
        hyp = comp["hypothesis"]

        print(f"\nSample {comp['sample_id']}:")
        print(f"REF: {ref[:150]}")
        print(f"HYP: {hyp[:150]}")
        print(f"REF words: {len(ref.split())}, HYP words: {len(hyp.split())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR model using greedy or n-gram LM decoding.")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model folder.",
    )

    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test CSV file.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug output for first few samples.",
    )

    parser.add_argument(
        "--segment_length",
        type=int,
        default=60,
        help="Segment length in seconds for pipeline evaluation.",
    )

    parser.add_argument(
        "--no_postprocess",
        action="store_true",
        help="Disable hypothesis post-processing during evaluation.",
    )

    parser.add_argument(
        "--decode_mode",
        choices=["greedy", "lm"],
        default="greedy",
        help="Decoding method: greedy CTC decoding or n-gram LM beam search decoding.",
    )

    args = parser.parse_args()

    config = Config()
    config.segment_length = args.segment_length

    if args.no_postprocess:
        print("[INFO] Hypothesis post-processing disabled for evaluation.")

        def no_postprocess(text: str) -> str:
            return str(text or "").upper().strip()

        globals()["apply_postprocess"] = no_postprocess

    test_csv = args.test_csv or os.path.join(config.base_dir, "test.csv")

    print("=" * 70)
    print(f"ASR EVALUATION - {args.decode_mode.upper()} DECODING")
    print("=" * 70)
    print(f"Model path     : {args.model_path}")
    print(f"Test CSV       : {test_csv}")
    print(f"Num samples    : {args.num_samples if args.num_samples is not None else 'ALL'}")
    print(f"Segment length : {config.segment_length}s")
    print(f"Decode mode    : {args.decode_mode}")
    print(f"Debug          : {args.debug}")
    print("=" * 70)

    print("\n1. Loading model...")
    model = ASRModel(config)

    model.load(
        args.model_path,
        use_lm=(args.decode_mode == "lm"),
    )

    print("\n2. Building pipeline...")
    preprocessor = AudioPreprocessor(config)
    pipeline = TranscriptionPipeline(model, preprocessor)

    print("\n3. Loading test data...")
    test_data = load_test_data(test_csv, args.num_samples)
    print(f"Loaded {len(test_data)} samples.")

    if not test_data:
        raise RuntimeError("No test data loaded. Check your CSV path and columns.")

    print("\n4. Running evaluation...")
    evaluator = evaluate(
        pipeline=pipeline,
        test_data=test_data,
        decode_mode=args.decode_mode,
        debug=args.debug,
    )

    print_results(evaluator, decode_mode=args.decode_mode)


if __name__ == "__main__":
    main()