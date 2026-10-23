"""
ClassScript - ASR Lecture Transcriber & AI Study Assistant
Native Gradio Interface optimized for Hugging Face ZeroGPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure src and src/web are on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "web"))
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import numpy as np
import wave

# ZeroGPU decorator with graceful local fallback
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=120)
    print("[ZeroGPU] spaces.GPU enabled with 120s duration quota")
except (ImportError, Exception):
    def gpu_decorator(fn):
        return fn
    print("[ZeroGPU] spaces not available, running in standard mode")

from config import WebConfig
from analysis import extract_keywords, extractive_summary, reading_time_label, word_count
from ai_service import (
    chat_about_transcript,
    generate_flashcards,
    generate_glossary,
    generate_quiz,
    is_ai_available,
    summarize_transcript,
)
from pipeline import Config, AudioPreprocessor, ASRModel, TranscriptionPipeline

_PIPELINE: TranscriptionPipeline | None = None


def get_pipeline() -> TranscriptionPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    model_path = WebConfig.MODEL_PATH
    p = Path(model_path)
    if not p.exists():
        raise RuntimeError(f"MODEL_PATH not found: {model_path}. Set HF_MODEL_REPO_ID or check model folder.")

    cfg = Config()
    cfg.augment = False
    model = ASRModel(cfg)
    model.load(str(model_path), use_lm=True)
    pre = AudioPreprocessor(cfg)
    _PIPELINE = TranscriptionPipeline(model, pre)
    return _PIPELINE


def convert_media_to_wav(input_path: str, output_path: str) -> str:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {res.stderr}")
    return output_path


def read_wav_pcm(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sampwidth == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = (v ^ 0x800000) - 0x800000
        x = v.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)

    return x.astype(np.float32), int(sr)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * target_peak
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def split_audio_into_chunks(audio: np.ndarray, sr: int, chunk_seconds: int = 30) -> List[Tuple[int, np.ndarray]]:
    chunk_samples = int(round(chunk_seconds * sr))
    total_samples = len(audio)
    if total_samples <= chunk_samples:
        return [(0, audio)]

    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append((start, audio[start:end]))
        start = end
    return chunks


def fmt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mm = int(seconds // 60)
    ss = int(seconds % 60)
    return f"{mm:02d}:{ss:02d}"


@gpu_decorator
def run_asr_inference(audio: np.ndarray, sr: int, decode_mode: str) -> Tuple[str, List[Dict[str, Any]]]:
    pipe = get_pipeline()
    duration_s = float(len(audio)) / float(sr)

    # Decode mode normalization
    mode = "lm" if "lm" in decode_mode.lower() else "greedy"

    if duration_s <= 60:
        raw_text = pipe.model.transcribe(audio, decode_mode=mode).strip()
        segments = [{"timestamp": "00:00", "seconds": 0.0, "text": raw_text}]
        return raw_text, segments

    # Chunked transcription for long audio
    chunks = split_audio_into_chunks(audio, sr=sr, chunk_seconds=30)
    text_parts = []
    segments = []

    for idx, (start_sample, chunk) in enumerate(chunks, start=1):
        start_seconds = float(start_sample) / float(sr)
        chunk_text = pipe.model.transcribe(chunk, decode_mode=mode).strip()
        if chunk_text:
            text_parts.append(chunk_text)
            segments.append({
                "timestamp": fmt_timestamp(start_seconds),
                "seconds": start_seconds,
                "text": chunk_text
            })

    full_text = " ".join(text_parts).strip()
    return full_text, segments


def transcribe_lecture(
    media_file: str,
    decode_mode: str,
    lecture_title: str,
    progress=gr.Progress(track_tqdm=True),
):
    if not media_file:
        raise gr.Error("Please upload an audio or video file first.")

    progress(0.1, desc="Converting media with FFmpeg...")
    temp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(temp_dir, "audio.wav")

    try:
        convert_media_to_wav(media_file, wav_path)
        audio, sr = read_wav_pcm(wav_path)
        audio = normalize_audio(audio)
        duration_s = float(len(audio)) / float(sr)
        duration_mins = duration_s / 60.0

        # Safety check: ZeroGPU has 120s execution budget
        if duration_mins > 35.0:
            raise gr.Error(
                f"Audio is {duration_mins:.1f} minutes long. For online ZeroGPU stability, "
                "please upload clips up to 30–35 minutes."
            )

        progress(0.3, desc=f"Running ASR ({'LM Beam Search' if 'lm' in decode_mode.lower() else 'Greedy CTC'})...")
        full_text, segments = run_asr_inference(audio, sr, decode_mode)

        progress(0.85, desc="Formatting output and analysis...")
        wc = word_count(full_text)
        reading_time = reading_time_label(wc)
        title = lecture_title.strip() or "Lecture Transcript"

        # Formatted markdown segments
        segment_lines = [f"### ⏱️ Timestamped Transcript — {title}\n"]
        for seg in segments:
            segment_lines.append(f"**`[{seg['timestamp']}]`** {seg['text']}")
        formatted_segments = "\n\n".join(segment_lines)

        stats_md = f"""
### 📊 Lecture Overview
- **Title**: {title}
- **Duration**: {fmt_timestamp(duration_s)} ({duration_mins:.1f} mins)
- **Word Count**: {wc:,} words
- **Reading Time**: ~{reading_time}
- **Decoding**: {'N-gram LM Beam Search' if 'lm' in decode_mode.lower() else 'Greedy CTC'}
"""

        # Export files
        txt_path = os.path.join(temp_dir, f"{title.replace(' ', '_')}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\nDuration: {fmt_timestamp(duration_s)}\n\n")
            f.write(formatted_segments)

        json_path = os.path.join(temp_dir, f"{title.replace(' ', '_')}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "title": title,
                "duration": fmt_timestamp(duration_s),
                "duration_seconds": round(duration_s, 1),
                "full_text": full_text,
                "segments": segments,
                "word_count": wc,
            }, f, indent=2, ensure_ascii=False)

        progress(1.0, desc="Done!")
        return (
            full_text,
            formatted_segments,
            stats_md,
            wav_path,
            txt_path,
            json_path,
            full_text,  # Updates transcript state
        )

    except Exception as e:
        raise gr.Error(f"Transcription failed: {str(e)}")


def ai_summary_handler(transcript: str, lecture_title: str):
    if not transcript or not transcript.strip():
        return "⚠️ Please transcribe a lecture in the Transcribe tab first."
    if not is_ai_available():
        return "⚠️ Gemini AI is not configured. Please set GEMINI_API_KEY in Space Secrets."
    try:
        summary = summarize_transcript(transcript, title=lecture_title)
        return summary
    except Exception as e:
        return f"⚠️ Error generating summary: {str(e)}"


def ai_quiz_handler(transcript: str, count: int, lecture_title: str):
    if not transcript or not transcript.strip():
        return "⚠️ Please transcribe a lecture in the Transcribe tab first."
    if not is_ai_available():
        return "⚠️ Gemini AI is not configured. Please set GEMINI_API_KEY in Space Secrets."
    try:
        res = generate_quiz(transcript, count=count, title=lecture_title)
        questions = res.get("questions", [])
        if not questions:
            return "No quiz questions could be generated."

        lines = [f"### 📝 Practice Quiz ({len(questions)} Questions)\n"]
        for idx, q in enumerate(questions, 1):
            lines.append(f"**Question {idx}:** {q['question']}\n")
            for opt in q["options"]:
                lines.append(f"- {opt}")
            lines.append(f"\n<details><summary><b>View Correct Answer & Explanation</b></summary>\n")
            lines.append(f"**Answer**: Option {q['answer']}\n\n**Explanation**: {q['explanation']}\n</details>\n---")

        return "\n\n".join(lines)
    except Exception as e:
        return f"⚠️ Error generating quiz: {str(e)}"


def ai_glossary_handler(transcript: str, lecture_title: str):
    if not transcript or not transcript.strip():
        return "⚠️ Please transcribe a lecture in the Transcribe tab first."
    if not is_ai_available():
        return "⚠️ Gemini AI is not configured. Please set GEMINI_API_KEY in Space Secrets."
    try:
        res = generate_glossary(transcript, title=lecture_title)
        terms = res.get("terms", [])
        if not terms:
            return "No glossary terms could be extracted."

        lines = [f"### 📚 Technical Terms Glossary\n"]
        for item in terms:
            lines.append(f"- **{item['term']}**: {item['definition']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"⚠️ Error extracting glossary: {str(e)}"


def ai_flashcards_handler(transcript: str, lecture_title: str):
    if not transcript or not transcript.strip():
        return "⚠️ Please transcribe a lecture in the Transcribe tab first."
    if not is_ai_available():
        return "⚠️ Gemini AI is not configured. Please set GEMINI_API_KEY in Space Secrets."
    try:
        res = generate_flashcards(transcript, count=8, title=lecture_title)
        cards = res.get("cards", [])
        if not cards:
            return "No flashcards could be generated."

        lines = [f"### 🗂️ Interactive Study Flashcards\n"]
        for idx, card in enumerate(cards, 1):
            lines.append(f"#### Card {idx}: {card['front']}")
            lines.append(f"<details><summary><b>Show Answer</b></summary>\n\n{card['back']}\n</details>\n---")
        return "\n\n".join(lines)
    except Exception as e:
        return f"⚠️ Error generating flashcards: {str(e)}"


def ai_chat_handler(message: str, history: List[Tuple[str, str]], transcript: str, lecture_title: str):
    if not message or not message.strip():
        return "", history
    if not transcript or not transcript.strip():
        bot_response = "⚠️ Please transcribe a lecture first so I have context to answer your questions."
        history = history + [(message, bot_response)]
        return "", history
    if not is_ai_available():
        bot_response = "⚠️ Gemini AI is not configured. Please set GEMINI_API_KEY in Space Secrets."
        history = history + [(message, bot_response)]
        return "", history

    try:
        ans = chat_about_transcript(
            transcript=transcript,
            question=message,
            title=lecture_title,
        )
        history = history + [(message, ans)]
        return "", history
    except Exception as e:
        history = history + [(message, f"⚠️ Error: {str(e)}")]
        return "", history


# -------------------------------------------------------------
# Gradio Portfolio UI
# -------------------------------------------------------------

CUSTOM_CSS = """
.main-header { text-align: center; margin-bottom: 1.5rem; }
.main-header h1 { font-size: 2.2rem; font-weight: 800; color: #4f46e5; margin-bottom: 0.2rem; }
.main-header p { font-size: 1.05rem; color: #6b7280; margin-bottom: 0.5rem; }
.badges-container { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.badge { display: inline-block; padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; background: #e0e7ff; color: #3730a3; }
.primary-btn { background: #4f46e5 !important; color: white !important; font-weight: 700 !important; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(title="ClassScript - ASR Lecture Transcriber & Study Assistant") as demo:
    # State storage for current transcript across tabs
    active_transcript = gr.State("")

    gr.HTML(f"""
    <style>
    {CUSTOM_CSS}
    </style>
    <div class="main-header">
        <h1>🎙️ ClassScript — ASR Lecture Transcriber</h1>
        <p>End-to-End Automatic Speech Recognition & AI Study Companion for Academic Lectures</p>
        <div class="badges-container">
            <span class="badge">Fine-Tuned Wav2Vec 2.0</span>
            <span class="badge">KenLM N-Gram LM Rescoring</span>
            <span class="badge">Google Gemini AI</span>
            <span class="badge">Hugging Face ZeroGPU</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # --- TAB 1: Transcription ---
        with gr.Tab("🎙️ Transcribe Lecture", id="tab-transcribe"):
            with gr.Row():
                with gr.Column(scale=1):
                    media_input = gr.File(
                        label="Upload Lecture Media (Audio or Video)",
                        file_types=["audio", "video"],
                    )
                    lecture_title_input = gr.Textbox(
                        label="Lecture Title (Optional)",
                        placeholder="e.g. CS101 - Introduction to Deep Learning",
                    )
                    decode_mode_input = gr.Radio(
                        choices=["N-gram LM Beam Search (Best Accuracy)", "Greedy CTC (Faster Baseline)"],
                        value="N-gram LM Beam Search (Best Accuracy)",
                        label="ASR Decoding Strategy",
                        info="LM Beam Search rescores predictions with an n-gram language model for lecture domain accuracy.",
                    )
                    transcribe_btn = gr.Button("✨ Start Transcription", variant="primary", size="lg")
                    gr.Markdown("💡 *Optimized for lecture clips up to 30–35 minutes on ZeroGPU.*")

                with gr.Column(scale=1):
                    stats_display = gr.Markdown("### 📊 Lecture Overview\n*Upload and transcribe a file to see metrics.*")
                    audio_player = gr.Audio(label="Audio Playback", type="filepath")

            with gr.Row():
                with gr.Column(scale=1):
                    transcript_output = gr.Textbox(
                        label="Full Text Transcript",
                        lines=12,
                        placeholder="Transcription will appear here...",
                    )
                    with gr.Row():
                        download_txt = gr.File(label="Download Transcript (.txt)")
                        download_json = gr.File(label="Download Transcript (.json)")

                with gr.Column(scale=1):
                    segments_output = gr.Markdown("### ⏱️ Timestamped Segments\n*Segments will appear here...*")

        # --- TAB 2: AI Summary ---
        with gr.Tab("📑 AI Summary", id="tab-summary"):
            gr.Markdown("Generate a structured academic summary with key points and core takeaways using Google Gemini.")
            gen_summary_btn = gr.Button("✨ Generate AI Summary", variant="primary")
            summary_display = gr.Markdown("*Transcribe a lecture first, then click Generate to create an AI summary.*")

        # --- TAB 3: Study Quiz ---
        with gr.Tab("📝 Practice Quiz", id="tab-quiz"):
            gr.Markdown("Auto-generate multiple-choice practice questions with answer keys and explanations grounded in lecture content.")
            with gr.Row():
                quiz_count_slider = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Number of Questions")
                gen_quiz_btn = gr.Button("✨ Generate Study Quiz", variant="primary")
            quiz_display = gr.Markdown("*Transcribe a lecture first, then click Generate to create practice questions.*")

        # --- TAB 4: Glossary & Flashcards ---
        with gr.Tab("📚 Glossary & Flashcards", id="tab-study"):
            with gr.Row():
                with gr.Column():
                    gen_glossary_btn = gr.Button("📖 Extract Key Terms & Glossary", variant="primary")
                    glossary_display = gr.Markdown("*Click button to extract key domain definitions.*")
                with gr.Column():
                    gen_flashcards_btn = gr.Button("🗂️ Generate Flashcards", variant="primary")
                    flashcards_display = gr.Markdown("*Click button to generate interactive revision flashcards.*")

        # --- TAB 5: Chat with Lecture (RAG) ---
        with gr.Tab("💬 Chat with Lecture", id="tab-chat"):
            gr.Markdown("Ask specific questions about the lecture content. Answers are grounded strictly in the transcript.")
            chatbot_ui = gr.Chatbot(label="Lecture Assistant", height=420)
            with gr.Row():
                chat_msg_input = gr.Textbox(
                    show_label=False,
                    placeholder="Ask a question about the lecture (e.g. 'What was the main conclusion?')...",
                    scale=4,
                )
                chat_send_btn = gr.Button("Send", variant="primary", scale=1)
            chat_clear_btn = gr.Button("Clear Chat History", size="sm")

    # -------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------
    transcribe_btn.click(
        fn=transcribe_lecture,
        inputs=[media_input, decode_mode_input, lecture_title_input],
        outputs=[
            transcript_output,
            segments_output,
            stats_display,
            audio_player,
            download_txt,
            download_json,
            active_transcript,
        ],
    )

    gen_summary_btn.click(
        fn=ai_summary_handler,
        inputs=[active_transcript, lecture_title_input],
        outputs=[summary_display],
    )

    gen_quiz_btn.click(
        fn=ai_quiz_handler,
        inputs=[active_transcript, quiz_count_slider, lecture_title_input],
        outputs=[quiz_display],
    )

    gen_glossary_btn.click(
        fn=ai_glossary_handler,
        inputs=[active_transcript, lecture_title_input],
        outputs=[glossary_display],
    )

    gen_flashcards_btn.click(
        fn=ai_flashcards_handler,
        inputs=[active_transcript, lecture_title_input],
        outputs=[flashcards_display],
    )

    chat_send_btn.click(
        fn=ai_chat_handler,
        inputs=[chat_msg_input, chatbot_ui, active_transcript, lecture_title_input],
        outputs=[chat_msg_input, chatbot_ui],
    )
    chat_msg_input.submit(
        fn=ai_chat_handler,
        inputs=[chat_msg_input, chatbot_ui, active_transcript, lecture_title_input],
        outputs=[chat_msg_input, chatbot_ui],
    )
    chat_clear_btn.click(lambda: [], outputs=[chatbot_ui])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    demo.launch(server_name=host, server_port=port, theme=theme)
