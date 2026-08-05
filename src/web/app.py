from __future__ import annotations

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import traceback
import gc
import sys
import time
import uuid
import wave
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from analysis import (
    extract_keywords,
    extractive_summary,
    reading_time_label,
    word_count,
)
from ai_service import (
    AINotConfiguredError,
    AIServiceError,
    ai_status,
    chat_about_transcript,
    generate_flashcards,
    generate_glossary,
    generate_quiz,
    is_ai_available,
    rag_answer,
    summarize_transcript,
    translate_segments,
    translate_transcript,
)
from export_service import build_docx, build_pdf
from rag_index import rebuild_index, search_lectures
from transcript_store import (
    delete_transcript,
    delete_transcripts,
    list_transcripts,
    load_notes,
    load_transcript,
    save_notes,
    save_transcript,
)

import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for, send_from_directory

from config import WebConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import Config, AudioPreprocessor, ASRModel, TranscriptionPipeline


app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(WebConfig)
app.secret_key = WebConfig.SECRET_KEY
WebConfig.init_app()

_PIPELINE: TranscriptionPipeline | None = None


def convert_media_to_wav(input_path: Path, output_path: Path) -> Path:
    """
    Convert uploaded audio/video media into 16kHz mono WAV using FFmpeg.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

    return output_path


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in WebConfig.ALLOWED_EXTENSIONS


def _read_wav_pcm(path: str) -> Tuple[np.ndarray, int]:
    """
    Read PCM WAV file and return float32 audio waveform and sampling rate.
    """
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
        v = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        v = (v ^ 0x800000) - 0x800000
        x = v.astype(np.float32) / 8388608.0

    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes.")

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)

    return x.astype(np.float32), int(sr)


def _resample_linear(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Lightweight fallback resampling using linear interpolation.
    """
    if sr == target_sr:
        return audio.astype(np.float32)

    if audio.size == 0:
        return audio.astype(np.float32)

    ratio = target_sr / float(sr)
    out_len = int(round(len(audio) * ratio))

    if out_len <= 1:
        return audio[:1].astype(np.float32)

    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float32)

    return np.interp(x_new, x_old, audio).astype(np.float32)


def _normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio signal to target peak amplitude.
    """
    if audio.size == 0:
        return audio.astype(np.float32)

    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)))

    if peak > 0:
        audio = audio / peak * target_peak

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def postprocess_transcript(input_text: str) -> str:
    return (input_text or "").strip()


def _get_pipeline() -> TranscriptionPipeline:
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    model_path = WebConfig.MODEL_PATH
    p = Path(model_path)

    if not p.exists():
        raise RuntimeError(f"MODEL_PATH not found: {model_path}")

    cfg = Config()
    cfg.augment = False

    model = ASRModel(cfg)
    model.load(model_path, use_lm=True)

    pre = AudioPreprocessor(cfg)
    _PIPELINE = TranscriptionPipeline(model, pre)

    return _PIPELINE


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        pass

    gc.collect()


def _split_audio_into_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_seconds: int = 10,
) -> List[Tuple[int, np.ndarray]]:
    """
    Split audio into fixed-duration chunks.
    """
    chunk_size = int(sr * chunk_seconds)
    min_chunk_size = int(sr * 0.5)

    chunks: List[Tuple[int, np.ndarray]] = []

    for start in range(0, len(audio), chunk_size):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        if len(chunk) >= min_chunk_size:
            chunks.append((start, chunk.astype(np.float32)))

    return chunks


def _decode_label(decode_mode: str) -> str:
    if decode_mode == "lm":
        return "N-gram LM Beam Search"
    return "Greedy CTC Decoding"


def transcribe_audio_direct(
    model: ASRModel,
    audio: np.ndarray,
    sr: int,
    decode_mode: str = "lm",
) -> Tuple[str, List[Dict[str, Any]]]:
    raw_text = model.transcribe(audio, decode_mode=decode_mode)
    raw_text = (raw_text or "").strip()

    segment_items: List[Dict[str, Any]] = []

    if raw_text:
        segment_items.append(
            {
                "timestamp_seconds": 0.0,
                "timestamp": "00:00",
                "text": postprocess_transcript(raw_text),
            }
        )

    return raw_text, segment_items


def transcribe_audio_in_chunks(
    pipe: TranscriptionPipeline,
    audio: np.ndarray,
    sr: int,
    chunk_seconds: int = 10,
    decode_mode: str = "lm",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Transcribe long audio in chunks using selected decoding mode.
    """
    chunks = _split_audio_into_chunks(
        audio=audio,
        sr=sr,
        chunk_seconds=chunk_seconds,
    )

    if not chunks:
        return "", []

    raw_text_parts: List[str] = []
    segment_items: List[Dict[str, Any]] = []

    total_chunks = len(chunks)

    for idx, (start_sample, chunk) in enumerate(chunks, start=1):
        start_seconds = float(start_sample) / float(sr)

        try:
            chunk_text = pipe.model.transcribe(chunk, decode_mode=decode_mode)
            chunk_text = (chunk_text or "").strip()

            if chunk_text:
                raw_text_parts.append(chunk_text)

                segment_items.append(
                    {
                        "timestamp_seconds": start_seconds,
                        "timestamp": _fmt_timestamp(start_seconds),
                        "text": postprocess_transcript(chunk_text),
                    }
                )

            print(f"Transcribed chunk {idx}/{total_chunks} using {_decode_label(decode_mode)}")

        except RuntimeError as e:
            error_msg = str(e).lower()
            _clear_cuda_cache()

            if "cuda out of memory" in error_msg or "out of memory" in error_msg:
                raise RuntimeError(
                    "CUDA out of memory while transcribing a chunk. "
                    "Please reduce chunk_seconds to 5 or 3."
                ) from e

            raise

        finally:
            _clear_cuda_cache()

    full_raw_text = " ".join(raw_text_parts).strip()
    return full_raw_text, segment_items


def _fmt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mm = int(seconds // 60)
    ss = int(seconds % 60)
    return f"{mm:02d}:{ss:02d}"


def _duration_str(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mm = int(seconds // 60)
    ss = int(seconds % 60)
    return f"{mm:02d}:{ss:02d}"


def cleanup_old_uploads() -> int:
    max_age_hours = int(getattr(WebConfig, "UPLOAD_MAX_AGE_HOURS", 0) or 0)
    if max_age_hours <= 0:
        return 0

    upload_dir = Path(WebConfig.UPLOAD_FOLDER)
    if not upload_dir.is_dir():
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0

    for path in upload_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass

    if removed:
        print(f"[cleanup] Removed {removed} upload file(s) older than {max_age_hours}h")

    return removed


def _make_segments(full_text: str, duration_s: float) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp_seconds": 0,
            "timestamp": _fmt_timestamp(0),
            "text": full_text.strip(),
        }
    ]


@app.get("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(WebConfig.UPLOAD_FOLDER, filename)


@app.get("/")
def index():
    cleanup_old_uploads()
    return render_template("index.html")


@app.post("/transcribe")
def transcribe():
    try:
        f = request.files.get("file")

        if not f or not f.filename:
            flash("No file selected.", "error")
            return redirect(url_for("index", upload=1))

        if not _allowed_file(f.filename):
            flash("Unsupported file type. Please upload audio or video media.", "error")
            return redirect(url_for("index", upload=1))

        lecture_title = (request.form.get("lecture_title") or "").strip() or "Lecture Transcript"
        course_code = (request.form.get("course_code") or "").strip()
        speaker = (request.form.get("speaker_name") or "").strip()

        decode_mode = (request.form.get("decode_mode") or "lm").strip().lower()

        if decode_mode not in {"greedy", "lm"}:
            decode_mode = "lm"

        print(f"[DEBUG] Selected decoding method: {_decode_label(decode_mode)}")

        job_id = uuid.uuid4().hex
        upload_dir = Path(WebConfig.UPLOAD_FOLDER)
        upload_dir.mkdir(parents=True, exist_ok=True)

        in_path = upload_dir / f"{job_id}_{Path(f.filename).name}"
        f.save(str(in_path))

        wav_path = upload_dir / f"{job_id}_converted.wav"

        if in_path.suffix.lower() == ".wav":
            audio_path = in_path
        else:
            audio_path = convert_media_to_wav(in_path, wav_path)

        audio_file_url = url_for("uploaded_file", filename=audio_path.name)

        audio, sr = _read_wav_pcm(str(audio_path))
        audio = _resample_linear(audio, sr, WebConfig.TARGET_SR)
        audio = _normalize_audio(audio)

        duration_s = float(len(audio)) / float(WebConfig.TARGET_SR)

        pipe = _get_pipeline()

        if duration_s <= 60:
            print(
                f"[DEBUG] Short audio ({duration_s:.1f}s): "
                f"using direct transcription with {_decode_label(decode_mode)}"
            )
            raw_text, chunk_segments = transcribe_audio_direct(
                model=pipe.model,
                audio=audio,
                sr=WebConfig.TARGET_SR,
                decode_mode=decode_mode,
            )
        else:
            print(
                f"[DEBUG] Long audio ({duration_s:.1f}s): "
                f"using chunked transcription with {_decode_label(decode_mode)}"
            )
            raw_text, chunk_segments = transcribe_audio_in_chunks(
                pipe=pipe,
                audio=audio,
                sr=WebConfig.TARGET_SR,
                chunk_seconds=WebConfig.CHUNK_SECONDS,
                decode_mode=decode_mode,
            )

        text = postprocess_transcript(raw_text)

        if chunk_segments:
            segments = chunk_segments
        else:
            segments = _make_segments(text, duration_s)

        wc = word_count(text)
        keywords = extract_keywords(text, top_n=WebConfig.KEYWORD_COUNT)
        summary = extractive_summary(text, num_sentences=WebConfig.SUMMARY_SENTENCES)

        transcript = {
            "job_id": job_id,
            "title": lecture_title,
            "course_code": course_code or None,
            "speaker": speaker or None,
            "decode_mode": _decode_label(decode_mode),
            "full_text": text,
            "segments": segments,
            "num_segments": len(segments),
            "duration": _duration_str(duration_s),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "audio_url": audio_file_url,
            "word_count": wc,
            "reading_time": reading_time_label(wc, wpm=WebConfig.READING_WPM),
            "keywords": keywords,
            "summary": summary,
        }

        save_transcript(job_id, transcript)
        rebuild_index()

        cleanup_old_uploads()
        flash("Transcription completed successfully.", "success")
        status = ai_status()
        return render_template(
            "result.html",
            transcript=transcript,
            ai_available=status["available"],
            ai_model=status.get("model", ""),
        )

    except Exception as e:
            traceback.print_exc()
            flash(f"Transcription failed: {e}", "error")
            return redirect(url_for("index", upload=1))


def _json_body() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _meta_from_body(body: Dict[str, Any]) -> Dict[str, str]:
    return {
        "title": (body.get("title") or "").strip(),
        "speaker": (body.get("speaker") or "").strip(),
        "course": (body.get("course") or "").strip(),
    }


def _require_transcript(body: Dict[str, Any]) -> str:
    transcript = (body.get("transcript") or "").strip()
    if not transcript:
        raise ValueError("transcript is required")
    return transcript


def _ai_response(handler):
    try:
        return jsonify(handler())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except AINotConfiguredError as e:
        return jsonify({"error": str(e)}), 503
    except AIServiceError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/ai/status")
def api_ai_status():
    return jsonify(ai_status())


@app.post("/api/ai/summarize")
def api_ai_summarize():
    body = _json_body()
    transcript = (body.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"error": "transcript is required"}), 400

    try:
        summary = summarize_transcript(
            transcript=transcript,
            title=(body.get("title") or "").strip(),
            speaker=(body.get("speaker") or "").strip(),
            course=(body.get("course") or "").strip(),
            style=(body.get("style") or "structured").strip(),
        )
        return jsonify({"summary": summary})
    except AINotConfiguredError as e:
        return jsonify({"error": str(e)}), 503
    except AIServiceError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {e}"}), 500


@app.post("/api/ai/chat")
def api_ai_chat():
    body = _json_body()
    transcript = (body.get("transcript") or "").strip()
    question = (body.get("question") or "").strip()
    history = body.get("history") or []

    if not transcript:
        return jsonify({"error": "transcript is required"}), 400
    if not question:
        return jsonify({"error": "question is required"}), 400

    if not isinstance(history, list):
        history = []

    try:
        answer = chat_about_transcript(
            transcript=transcript,
            question=question,
            history=history,
            title=(body.get("title") or "").strip(),
            speaker=(body.get("speaker") or "").strip(),
            course=(body.get("course") or "").strip(),
        )
        return jsonify({"answer": answer})
    except AINotConfiguredError as e:
        return jsonify({"error": str(e)}), 503
    except AIServiceError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Chat failed: {e}"}), 500


@app.post("/api/ai/quiz")
def api_ai_quiz():
    body = _json_body()

    def run():
        t = _require_transcript(body)
        data = generate_quiz(
            t,
            count=int(body.get("count") or 5),
            **_meta_from_body(body),
        )
        return data

    return _ai_response(run)


@app.post("/api/ai/glossary")
def api_ai_glossary():
    body = _json_body()

    def run():
        t = _require_transcript(body)
        return generate_glossary(t, max_terms=int(body.get("max_terms") or 15), **_meta_from_body(body))

    return _ai_response(run)


@app.post("/api/ai/flashcards")
def api_ai_flashcards():
    body = _json_body()

    def run():
        t = _require_transcript(body)
        return generate_flashcards(t, count=int(body.get("count") or 10), **_meta_from_body(body))

    return _ai_response(run)


@app.post("/api/ai/translate")
def api_ai_translate():
    body = _json_body()

    def run():
        lang = (body.get("target_language") or "Urdu").strip()
        segments = body.get("segments")
        if isinstance(segments, list) and segments:
            translated_segments = translate_segments(segments, lang)
            paragraph = "\n\n".join((s.get("text") or "").strip() for s in translated_segments if s.get("text"))
            return {
                "translation": paragraph,
                "translated_segments": translated_segments,
                "target_language": lang,
            }

        t = _require_transcript(body)
        translation = translate_transcript(t, lang, **_meta_from_body(body))
        return {"translation": translation, "target_language": lang}

    return _ai_response(run)


@app.get("/api/library")
def api_library_list():
    return jsonify({"lectures": list_transcripts()})


@app.delete("/api/library/<job_id>")
def api_library_delete(job_id: str):
    deleted = delete_transcript(job_id)
    if not deleted:
        return jsonify({"error": "lecture not found", "job_id": job_id}), 404
    return jsonify({"ok": True, "job_id": job_id})


@app.post("/api/library/bulk-delete")
def api_library_bulk_delete():
    body = _json_body()
    job_ids = body.get("job_ids")
    if not isinstance(job_ids, list):
        return jsonify({"error": "job_ids must be a list"}), 400

    cleaned_ids = [str(x).strip() for x in job_ids if str(x).strip()]
    result = delete_transcripts(cleaned_ids)
    return jsonify(result)


@app.post("/api/rag/search")
def api_rag_search():
    body = _json_body()
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    results = search_lectures(query, top_k=int(body.get("top_k") or 8))
    return jsonify({"results": results})


@app.post("/api/rag/ask")
def api_rag_ask():
    body = _json_body()
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    def run():
        hits = search_lectures(question, top_k=int(body.get("top_k") or 6))
        answer = rag_answer(question, hits)
        return {"answer": answer, "sources": hits}

    return _ai_response(run)


@app.get("/library")
def library_page():
    return render_template(
        "library.html",
        lectures=list_transcripts(),
        ai_available=is_ai_available(),
    )


@app.get("/transcript/<job_id>")
def view_transcript(job_id: str):
    transcript = load_transcript(job_id)
    if transcript is None:
        flash("Transcript not found.", "error")
        return redirect(url_for("library_page"))
    status = ai_status()
    return render_template(
        "result.html",
        transcript=transcript,
        ai_available=status["available"],
        ai_model=status.get("model", ""),
    )


@app.get("/api/notes/<job_id>")
def api_get_notes(job_id: str):
    return jsonify({"job_id": job_id, "notes": load_notes(job_id)})


@app.post("/api/notes/<job_id>")
def api_save_notes(job_id: str):
    body = _json_body()
    notes = body.get("notes")
    if not isinstance(notes, dict):
        return jsonify({"error": "notes must be an object"}), 400
    cleaned = {str(k): str(v) for k, v in notes.items()}
    save_notes(job_id, cleaned)
    return jsonify({"ok": True, "job_id": job_id})


@app.post("/api/export/docx")
def api_export_docx():
    body = _json_body()
    transcript = body.get("transcript")
    if not isinstance(transcript, dict):
        return jsonify({"error": "transcript object is required"}), 400
    try:
        import io

        data = build_docx(transcript, extras=body.get("extras") or {})
        title = (transcript.get("title") or "transcript").replace(" ", "_")[:40]
        return send_file(
            io.BytesIO(data),
            as_attachment=True,
            download_name=f"{title}.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except ImportError:
        return jsonify({"error": "Install python-docx: pip install python-docx"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/export/pdf")
def api_export_pdf():
    body = _json_body()
    transcript = body.get("transcript")
    if not isinstance(transcript, dict):
        return jsonify({"error": "transcript object is required"}), 400
    try:
        import io

        data = build_pdf(transcript, extras=body.get("extras") or {})
        if isinstance(data, bytearray):
            data = bytes(data)
        elif isinstance(data, str):
            data = data.encode("utf-8")
        title = (transcript.get("title") or "transcript").replace(" ", "_")[:40]
        return send_file(
            io.BytesIO(data),
            as_attachment=True,
            download_name=f"{title}.pdf",
            mimetype="application/pdf",
        )
    except ImportError:
        return jsonify({"error": "Install fpdf2: pip install fpdf2"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting ASR Web Interface...")
    print(f"Upload folder: {WebConfig.UPLOAD_FOLDER}")
    print(f"Model path: {WebConfig.MODEL_PATH}")
    cleanup_old_uploads()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=True,
    )
