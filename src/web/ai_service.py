"""
Optional LLM layer for transcript Q&A and summarization.

This module is independent of the Wav2Vec2 / ASR pipeline. It calls an
OpenAI-compatible chat API (OpenAI, Ollama, Groq, etc.) when configured.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# Load .env from web folder when python-dotenv is installed.
try:
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


class AINotConfiguredError(RuntimeError):
    """Raised when no LLM API is configured."""


class AIServiceError(RuntimeError):
    """Raised when the LLM API returns an error."""


def _api_key() -> str:
    return (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()


def _api_base() -> str:
    base = (os.environ.get("LLM_API_BASE") or "https://api.openai.com/v1").strip()
    return base.rstrip("/")


def _model() -> str:
    return (os.environ.get("LLM_MODEL") or "gpt-4o-mini").strip()


def _max_transcript_chars() -> int:
    try:
        return max(2000, int(os.environ.get("LLM_MAX_TRANSCRIPT_CHARS", "14000")))
    except ValueError:
        return 14000


def is_ai_available() -> bool:
    """True when an API key is set (use any value for local Ollama)."""
    return bool(_api_key())


def ai_status() -> Dict[str, Any]:
    return {
        "available": is_ai_available(),
        "model": _model(),
        "provider_hint": "Set OPENAI_API_KEY or LLM_API_KEY in .env (see .env.example)",
    }


def truncate_transcript(text: str, max_chars: Optional[int] = None) -> str:
    max_chars = max_chars or _max_transcript_chars()
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[... middle of transcript omitted for length ...]\n\n"
        + text[-half:]
    )


def _chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    timeout: int = 120,
) -> str:
    if not is_ai_available():
        raise AINotConfiguredError(
            "AI features are not configured. Add OPENAI_API_KEY or LLM_API_KEY to src/web/.env"
        )

    url = f"{_api_base()}/chat/completions"
    payload = {
        "model": _model(),
        "messages": messages,
        "temperature": temperature,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise AIServiceError(f"LLM API error ({e.code}): {detail[:500]}") from e
    except urllib.error.URLError as e:
        reason = str(e.reason) if getattr(e, "reason", None) else str(e)
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            raise AIServiceError(
                "Request timed out. For long lectures, translation runs in chunks—please try again."
            ) from e
        raise AIServiceError(f"Could not reach LLM API at {_api_base()}: {e}") from e

    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise AIServiceError(f"Unexpected LLM response: {data!r}") from e


def _lecture_context_block(
    transcript: str,
    title: str = "",
    speaker: str = "",
    course: str = "",
) -> str:
    meta = []
    if title:
        meta.append(f"Title: {title}")
    if course:
        meta.append(f"Course: {course}")
    if speaker:
        meta.append(f"Speaker: {speaker}")

    header = "\n".join(meta)
    body = truncate_transcript(transcript)
    if header:
        return f"{header}\n\n--- TRANSCRIPT ---\n{body}"
    return f"--- TRANSCRIPT ---\n{body}"


def summarize_transcript(
    transcript: str,
    title: str = "",
    speaker: str = "",
    course: str = "",
    style: str = "structured",
) -> str:
    """
    Generate an AI summary of the full lecture transcript.
    """
    context = _lecture_context_block(transcript, title, speaker, course)

    style_instruction = (
        "Use clear sections: Overview, Key Points (bullet list with hyphen - only, never +), and Conclusion."
        if style == "structured"
        else "Write a concise paragraph summary."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic lecture assistant. Summarize the lecture transcript "
                "accurately. Do not invent facts not present in the transcript. "
                + style_instruction
            ),
        },
        {
            "role": "user",
            "content": f"Summarize this lecture:\n\n{context}",
        },
    ]

    return _chat_completion(messages, temperature=0.2)


def chat_about_transcript(
    transcript: str,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    title: str = "",
    speaker: str = "",
    course: str = "",
) -> str:
    """
    Answer a question using only the provided transcript as context.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is required.")

    context = _lecture_context_block(transcript, title, speaker, course)

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You help students understand a lecture using ONLY the transcript below. "
                "Rules:\n"
                "- Answer from the transcript; cite timestamps if segments were provided.\n"
                "- If the answer is not in the transcript, say you cannot find it in the lecture.\n"
                "- Be clear and educational; keep answers focused.\n\n"
                f"{context}"
            ),
        },
    ]

    for msg in history or []:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})

    return _chat_completion(messages, temperature=0.4)


def _parse_json_from_llm(raw: str) -> Any:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidate = raw[start : end + 1]
        else:
            raise AIServiceError("LLM did not return valid JSON.")

    # Repair common JSON mistakes from LLMs.
    candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    # Missing commas between string literals:  "a" "b"  -> "a","b"
    candidate = re.sub(r"\"\s*\"", "\",\"", candidate)
    # Missing commas between objects/arrays.
    candidate = re.sub(r"}\s*{", "},{", candidate)
    candidate = re.sub(r"]\s*\[", "],[", candidate)
    candidate = re.sub(r"\n+", "\n", candidate).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise AIServiceError(f"LLM JSON parse failed: {e}") from e


def generate_quiz(
    transcript: str,
    count: int = 5,
    title: str = "",
    speaker: str = "",
    course: str = "",
) -> Dict[str, Any]:
    context = _lecture_context_block(transcript, title, speaker, course)
    messages = [
        {
            "role": "system",
            "content": (
                "Create multiple-choice quiz questions from the lecture transcript only.\n"
                "Output MUST use this exact plain-text format for each question:\n"
                "Q: <question>\n"
                "A) <option>\n"
                "B) <option>\n"
                "C) <option>\n"
                "D) <option>\n"
                "ANSWER: <A|B|C|D>\n"
                "EXPLANATION: <short explanation>\n"
                "---\n"
                "Do not output JSON. Do not include extra commentary."
            ),
        },
        {"role": "user", "content": f"Create {count} MCQs:\n\n{context}"},
    ]
    raw = _chat_completion(messages, temperature=0.35)
    blocks = [b.strip() for b in raw.split("---") if b.strip()]
    questions: List[Dict[str, Any]] = []

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        q = ""
        options = {"A": "", "B": "", "C": "", "D": ""}
        answer = "A"
        explanation = ""

        for ln in lines:
            upper = ln.upper()
            if upper.startswith("Q:"):
                q = ln.split(":", 1)[1].strip()
            elif re.match(r"^[A-D]\)\s*", ln, flags=re.I):
                letter = ln[0].upper()
                options[letter] = re.sub(r"^[A-D]\)\s*", "", ln, flags=re.I).strip()
            elif upper.startswith("ANSWER:"):
                val = ln.split(":", 1)[1].strip().upper()
                answer = val[:1] if val and val[0] in "ABCD" else "A"
            elif upper.startswith("EXPLANATION:"):
                explanation = ln.split(":", 1)[1].strip()

        if q and all(options[k] for k in ("A", "B", "C", "D")):
            questions.append(
                {
                    "question": q,
                    "options": [f"A) {options['A']}", f"B) {options['B']}", f"C) {options['C']}", f"D) {options['D']}"],
                    "answer": answer,
                    "explanation": explanation or "Based on the lecture content.",
                }
            )

    if not questions:
        raise AIServiceError("Could not parse quiz response from LLM.")

    return {"questions": questions[:count]}


def generate_glossary(
    transcript: str,
    max_terms: int = 15,
    title: str = "",
    speaker: str = "",
    course: str = "",
) -> Dict[str, Any]:
    context = _lecture_context_block(transcript, title, speaker, course)
    messages = [
        {
            "role": "system",
            "content": (
                "Extract important technical terms from the lecture and define them clearly. "
                'Return ONLY JSON: {"terms":[{"term":"...","definition":"..."}]}'
            ),
        },
        {"role": "user", "content": f"Up to {max_terms} terms:\n\n{context}"},
    ]
    data = _parse_json_from_llm(_chat_completion(messages, temperature=0.25))
    if not isinstance(data, dict) or "terms" not in data:
        raise AIServiceError("Invalid glossary JSON from LLM.")
    return data


def generate_flashcards(
    transcript: str,
    count: int = 10,
    title: str = "",
    speaker: str = "",
    course: str = "",
) -> Dict[str, Any]:
    context = _lecture_context_block(transcript, title, speaker, course)
    messages = [
        {
            "role": "system",
            "content": (
                "Create study flashcards from the lecture. "
                'Return ONLY JSON: {"cards":[{"front":"...","back":"..."}]}'
            ),
        },
        {"role": "user", "content": f"Create {count} flashcards:\n\n{context}"},
    ]
    data = _parse_json_from_llm(_chat_completion(messages, temperature=0.35))
    if not isinstance(data, dict) or "cards" not in data:
        raise AIServiceError("Invalid flashcards JSON from LLM.")
    return data


def _split_text_chunks(text: str, max_chars: int = 2800) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        part_len = len(paragraph) + 1
        if current_len + part_len > max_chars and current:
            chunks.append("\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += part_len

    if current:
        chunks.append("\n".join(current))

    if not chunks:
        return [text[:max_chars]]
    return chunks


def _translate_chunk(chunk: str, target_language: str) -> str:
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return ""

    numbered = "\n".join(f"L{i+1}: {ln}" for i, ln in enumerate(lines))

    messages = [
        {
            "role": "system",
            "content": (
                f"Translate each line to {target_language}.\n"
                "CRITICAL RULES:\n"
                "- Do NOT summarize.\n"
                "- Translate every line.\n"
                "- Keep numbering exactly: L1:, L2:, ...\n"
                "- Do not add or remove lines.\n"
                "- Output only translated numbered lines."
            ),
        },
        {"role": "user", "content": numbered},
    ]
    raw = _chat_completion(messages, temperature=0.0, timeout=240)
    out_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    translated: List[str] = []
    for ln in out_lines:
        m = re.match(r"^L\d+:\s*(.*)$", ln)
        if m:
            translated.append(m.group(1).strip())

    # If model ignored numbering, fail this chunk so caller can retry/fallback.
    if not translated:
        raise AIServiceError("Translation chunk did not preserve line mapping.")

    return "\n".join(translated).strip()


def _lang_to_code(name: str) -> str:
    mapping = {
        "urdu": "ur",
        "hindi": "hi",
        "english": "en",
        "arabic": "ar",
        "french": "fr",
        "spanish": "es",
        "malay": "ms",
    }
    return mapping.get((name or "").strip().lower(), "en")


def _try_machine_translate(text: str, target_language: str) -> Optional[str]:
    """
    Prefer deterministic translation if deep_translator is installed.
    """
    try:
        from deep_translator import GoogleTranslator  # type: ignore
    except Exception:
        return None

    code = _lang_to_code(target_language)
    source_text = (text or "").strip()
    if not source_text:
        return ""

    def split_for_provider(s: str, max_chars: int = 4200) -> List[str]:
        # Keep each provider call comfortably below 5000-char limits.
        if len(s) <= max_chars:
            return [s]
        chunks: List[str] = []
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", s):
            if not sentence:
                continue
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    part = sentence[i:i + max_chars]
                    if part:
                        chunks.append(part)
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > max_chars:
                if current:
                    chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [s[:max_chars]]

    try:
        translator = GoogleTranslator(source="auto", target=code)
        chunks = split_for_provider(source_text, max_chars=4200)
        translated_parts: List[str] = []
        for chunk in chunks:
            translated_parts.append(translator.translate(chunk))
        return "\n\n".join(p for p in translated_parts if p)
    except Exception:
        # Any provider-specific limitation/error falls back to LLM path.
        return None


def translate_transcript(
    transcript: str,
    target_language: str,
    title: str = "",
    speaker: str = "",
    course: str = "",
) -> str:
    """
    Translate long transcripts in chunks to avoid API timeouts.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return ""

    machine = _try_machine_translate(transcript, target_language)
    if machine and machine.strip():
        return machine.strip()

    chunks = _split_text_chunks(transcript, max_chars=2800)
    if not chunks:
        return ""

    translated_parts: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prefix = ""
        if idx == 1 and title:
            prefix = f"[{title}]\n\n"
        try:
            part = _translate_chunk(chunk, target_language)
        except AIServiceError:
            # Retry once with smaller chunk for providers that drop formatting.
            mini_chunks = _split_text_chunks(chunk, max_chars=1200)
            part = "\n".join(_translate_chunk(mc, target_language) for mc in mini_chunks if mc.strip())
        if part:
            translated_parts.append(prefix + part if prefix else part)
    translated = "\n\n".join(translated_parts)
    return translated


def translate_segments(
    segments: List[Dict[str, Any]],
    target_language: str,
) -> List[Dict[str, Any]]:
    """
    Translate timestamped segments while preserving timestamps.
    """
    out: List[Dict[str, Any]] = []
    for seg in segments or []:
        text = (seg.get("text") or "").strip()
        translated = translate_transcript(text, target_language) if text else ""
        out.append(
            {
                "timestamp": seg.get("timestamp", ""),
                "timestamp_seconds": seg.get("timestamp_seconds", 0),
                "text": translated or text,
            }
        )
    return out


def rag_answer(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "No saved lectures matched your question. Transcribe more lectures first."

    context_parts = []
    for ch in retrieved_chunks:
        context_parts.append(
            f"Lecture: {ch.get('title')} [{ch.get('timestamp')}]\n{ch.get('text')}"
        )
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer using ONLY the excerpts from saved lectures below. "
                "Mention which lecture/timestamp when relevant. "
                "If not covered, say so.\n\n" + context
            ),
        },
        {"role": "user", "content": question},
    ]
    return _chat_completion(messages, temperature=0.35)
