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
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from web folder when python-dotenv is installed.
def _reload_env() -> None:
    try:
        from dotenv import load_dotenv
        env_file = Path(__file__).resolve().parent / ".env"
        if env_file.is_file():
            raw = env_file.read_text(encoding="utf-8")
            if "LLM_API_BASE" not in raw:
                os.environ.pop("LLM_API_BASE", None)
            load_dotenv(env_file, override=True)
    except Exception:
        pass

_reload_env()


class AINotConfiguredError(RuntimeError):
    """Raised when no LLM API is configured."""


class AIServiceError(RuntimeError):
    """Raised when the LLM API returns an error."""


def _api_key() -> str:
    _reload_env()
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()


def _model() -> str:
    _reload_env()
    m = (os.environ.get("GEMINI_MODEL") or os.environ.get("LLM_MODEL") or "").strip()
    if m:
        if m in {"gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-pro"}:
            return "gemini-3.5-flash"
        return m
    return "gemini-3.5-flash"


def _provider() -> str:
    _reload_env()
    prov = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if prov:
        return prov
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if _model().startswith("gemini"):
        return "gemini"
    key = _api_key()
    if key.startswith("AIzaSy"):
        return "gemini"
    base = (os.environ.get("LLM_API_BASE") or "").lower()
    if "googleapis" in base or "google" in base or "gemini" in base:
        return "gemini"
    if "localhost:11434" in base or "127.0.0.1:11434" in base:
        return "ollama"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "gemini"


def _api_base() -> str:
    _reload_env()
    base = os.environ.get("LLM_API_BASE")
    if base and base.strip():
        return base.strip().rstrip("/")
    if _provider() == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta/openai"
    if _provider() == "ollama":
        return "http://localhost:11434/v1"
    return "https://api.openai.com/v1"


def _max_transcript_chars() -> int:
    try:
        return max(2000, int(os.environ.get("LLM_MAX_TRANSCRIPT_CHARS", "14000")))
    except ValueError:
        return 14000


def is_ai_available() -> bool:
    """True when an API key is set and not a placeholder."""
    key = _api_key()
    if not key:
        return False
    placeholders = {
        "your-gemini-api-key-here",
        "your_gemini_api_key_here",
        "your-key-here",
        "sk-your-key-here",
        "ollama",
    }
    if key.lower() in placeholders:
        return False
    return True


def ai_status() -> Dict[str, Any]:
    return {
        "available": is_ai_available(),
        "model": _model(),
        "provider": _provider(),
        "provider_hint": (
            f"Ready — {_model()}"
            if is_ai_available()
            else "Add GEMINI_API_KEY to src/web/.env (get key at https://aistudio.google.com/apikey)"
        ),
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


def _call_gemini_native(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    timeout: int = 120,
    response_format: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> str:
    """
    Direct call to Google Gemini generateContent REST endpoint.
    Used as an alternate or fallback for native Gemini requests.
    """
    key = _api_key()
    selected_model = model or _model()
    if not selected_model.startswith("gemini-") or selected_model in {"gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-pro"}:
        selected_model = "gemini-3.5-flash"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={key}"

    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        else:
            contents.append({"role": "user", "parts": [{"text": text}]})

    if not contents and system_parts:
        contents.append({"role": "user", "parts": [{"text": "\n\n".join(system_parts)}]})
        system_parts = []

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
        },
    }
    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }
    if response_format and response_format.get("type") == "json_object":
        payload["generationConfig"]["responseMimeType"] = "application/json"

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(detail)
            msg = err_json.get("error", {}).get("message", detail[:300])
        except Exception:
            msg = detail[:300]
        if "API_KEY_INVALID" in detail or "valid API key" in msg.lower():
            raise AIServiceError(
                "Invalid Gemini API key. Please check GEMINI_API_KEY in src/web/.env."
            ) from e
        raise AIServiceError(f"Gemini API error ({e.code}): {msg}") from e
    except urllib.error.URLError as e:
        reason = str(e.reason) if getattr(e, "reason", None) else str(e)
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            raise AIServiceError("Gemini request timed out. Please try again.") from e
        raise AIServiceError(f"Could not reach Gemini API: {reason}") from e


def _chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    timeout: int = 120,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    if not is_ai_available():
        raise AINotConfiguredError(
            "AI features are not configured. "
            "Please add GEMINI_API_KEY to src/web/.env (get a free key at https://aistudio.google.com/apikey)"
        )

    base = _api_base()
    primary_model = _model()
    candidate_models = [primary_model]
    if _provider() == "gemini":
        for fb in ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]:
            if fb not in candidate_models:
                candidate_models.append(fb)

    last_error: Optional[Exception] = None

    for model_name in candidate_models:
        url = f"{base}/chat/completions"
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

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
            return (data["choices"][0]["message"]["content"] or "").strip()

        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(detail)
                if isinstance(err_json, list) and err_json and "error" in err_json[0]:
                    msg = err_json[0]["error"].get("message", "")
                elif isinstance(err_json, dict) and "error" in err_json:
                    msg = err_json["error"].get("message", "")
                else:
                    msg = detail[:300]
            except Exception:
                msg = detail[:300]

            if "API key not valid" in msg or "API_KEY_INVALID" in msg or "valid API key" in msg.lower():
                raise AIServiceError(
                    "Invalid Gemini API key. Please check GEMINI_API_KEY in src/web/.env "
                    "(get a free key at https://aistudio.google.com/apikey)."
                ) from e

            # If 404, 429 or 503, retry with next candidate model
            if e.code in (404, 429, 503):
                last_error = AIServiceError(f"AI API error ({e.code}): {msg}")
                continue

            # Try native endpoint fallback for Gemini
            if _provider() == "gemini":
                try:
                    return _call_gemini_native(
                        messages,
                        temperature=temperature,
                        timeout=timeout,
                        response_format=response_format,
                        model=model_name,
                    )
                except Exception:
                    pass

            raise AIServiceError(f"AI API error ({e.code}): {msg}") from e

        except urllib.error.URLError as e:
            reason = str(e.reason) if getattr(e, "reason", None) else str(e)
            if "timed out" in reason.lower() or "timeout" in reason.lower():
                last_error = AIServiceError("Request timed out. Please try again.")
                continue

            if "10061" in reason or "refused" in reason.lower():
                raise AIServiceError(
                    f"Could not connect to LLM service at {base}. "
                    "If using Google Gemini, set GEMINI_API_KEY in src/web/.env."
                ) from e

            if _provider() == "gemini":
                try:
                    return _call_gemini_native(
                        messages,
                        temperature=temperature,
                        timeout=timeout,
                        response_format=response_format,
                        model=model_name,
                    )
                except Exception:
                    pass

            raise AIServiceError(f"Could not reach AI API ({base}): {reason}") from e

        except (KeyError, IndexError, TypeError) as e:
            if _provider() == "gemini":
                try:
                    return _call_gemini_native(
                        messages,
                        temperature=temperature,
                        timeout=timeout,
                        response_format=response_format,
                        model=model_name,
                    )
                except Exception:
                    pass
            raise AIServiceError(f"Unexpected AI response format: {e}") from e

    # If all OpenAI-compatible attempts failed and provider is gemini, try native endpoint
    if _provider() == "gemini":
        for native_m in ["gemini-3.5-flash", "gemini-3.1-flash-lite"]:
            try:
                return _call_gemini_native(
                    messages,
                    temperature=temperature,
                    timeout=timeout,
                    response_format=response_format,
                    model=native_m,
                )
            except Exception as e:
                last_error = e
                continue

    if last_error:
        raise last_error

    raise AIServiceError("All candidate AI models failed. Please try again shortly.")


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
    """
    Parse JSON returned by an LLM.

    Handles:
    - Normal JSON
    - ```json ... ``` code fences
    - Extra text before/after JSON
    - Smart quotes
    - Trailing commas
    - Python-style single-quoted dictionaries/lists
    """

    raw = (raw or "").strip()

    if not raw:
        raise AIServiceError("LLM returned an empty response.")

    # ---------------------------------------------------------
    # 1. Remove Markdown code fences
    # ---------------------------------------------------------
    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines:
            # Remove opening ```json / ```
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    # ---------------------------------------------------------
    # 2. Try normal JSON first
    # ---------------------------------------------------------
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # ---------------------------------------------------------
    # 3. Extract JSON object from surrounding text
    # ---------------------------------------------------------
    start = raw.find("{")
    end = raw.rfind("}")

    if start >= 0 and end > start:
        candidate = raw[start:end + 1]
    else:
        candidate = raw

    # ---------------------------------------------------------
    # 4. Normalize common LLM formatting mistakes
    # ---------------------------------------------------------
    candidate = (
        candidate
        .replace("\ufeff", "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("’", "'")
    )

    # Remove trailing commas:
    # {"a": 1,} -> {"a": 1}
    # [1, 2,]   -> [1, 2]
    candidate = re.sub(
        r",\s*([}\]])",
        r"\1",
        candidate,
    )

    # Fix missing commas between objects:
    # } { -> }, {
    candidate = re.sub(
        r"}\s*{",
        "},{",
        candidate,
    )

    # Fix missing commas between arrays:
    # ] [ -> ], [
    candidate = re.sub(
        r"]\s*\[",
        "],[",
        candidate,
    )

    candidate = candidate.strip()

    # ---------------------------------------------------------
    # 5. Try JSON again after repairs
    # ---------------------------------------------------------
    try:
        return json.loads(candidate)

    except json.JSONDecodeError:
        pass

    # ---------------------------------------------------------
    # 6. Last fallback:
    #    Llama models sometimes return Python-style dictionaries
    #    using single quotes.
    #
    #    ast.literal_eval is safe for Python literals and does
    #    not execute arbitrary code.
    # ---------------------------------------------------------
    try:
        import ast

        parsed = ast.literal_eval(candidate)

        if isinstance(parsed, (dict, list)):
            return parsed

    except (ValueError, SyntaxError, TypeError):
        pass

    # ---------------------------------------------------------
    # 7. Give a useful error
    # ---------------------------------------------------------
    try:
        json.loads(candidate)
    except json.JSONDecodeError as e:
        preview = candidate[:500].replace("\n", "\\n")

        raise AIServiceError(
            f"LLM JSON parse failed: {e}. "
            f"Response preview: {preview}"
        ) from e

    raise AIServiceError("LLM returned an unsupported JSON format.")


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

    context = _lecture_context_block(
        transcript,
        title,
        speaker,
        course,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic lecture assistant.\n\n"

                "Extract the most important technical terms "
                "from the lecture transcript.\n\n"

                "For every term, provide a clear and concise "
                "definition based ONLY on the lecture transcript.\n\n"

                "Return ONLY valid JSON.\n\n"

                'The exact required structure is:\n'
                '{"terms":[{"term":"example","definition":"example definition"}]}\n\n'

                "Rules:\n"
                f"- Return no more than {max_terms} terms.\n"
                "- Every item must contain exactly a term and definition.\n"
                "- Use double quotes for JSON strings.\n"
                "- Do not use Markdown.\n"
                "- Do not use ```json.\n"
                "- Do not add explanations before or after the JSON.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Extract up to {max_terms} important technical terms "
                f"from this lecture:\n\n{context}"
            ),
        },
    ]

    raw = _chat_completion(
        messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    data = _parse_json_from_llm(raw)

    # ---------------------------------------------------------
    # Validate top-level structure
    # ---------------------------------------------------------
    if not isinstance(data, dict):
        raise AIServiceError(
            "Invalid glossary response: expected a JSON object."
        )

    terms = data.get("terms")

    if not isinstance(terms, list):
        raise AIServiceError(
            "Invalid glossary response: 'terms' must be a list."
        )

    # ---------------------------------------------------------
    # Clean and validate individual terms
    # ---------------------------------------------------------
    cleaned_terms: List[Dict[str, str]] = []

    for item in terms:
        if not isinstance(item, dict):
            continue

        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()

        if not term or not definition:
            continue

        cleaned_terms.append(
            {
                "term": term,
                "definition": definition,
            }
        )

    if not cleaned_terms:
        raise AIServiceError(
            "The LLM returned no valid glossary terms."
        )

    return {
        "terms": cleaned_terms[:max_terms]
    }


def generate_flashcards(
    transcript: str,
    count: int = 10,
    title: str = "",
    speaker: str = "",
    course: str = "",
) -> Dict[str, Any]:

    context = _lecture_context_block(
        transcript,
        title,
        speaker,
        course,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic lecture assistant.\n\n"
                "Create study flashcards from the lecture transcript.\n"
                "Use ONLY information from the transcript.\n\n"

                "Return ONLY valid JSON.\n\n"

                'The exact required structure is:\n'
                '{"cards":[{"front":"question or term","back":"answer or explanation"}]}\n\n'

                "Rules:\n"
                f"- Create no more than {count} cards.\n"
                "- Every card must contain front and back.\n"
                "- Use double quotes for JSON strings.\n"
                "- Do not use Markdown.\n"
                "- Do not use ```json.\n"
                "- Do not add explanations before or after the JSON.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Create {count} study flashcards:\n\n{context}"
            ),
        },
    ]

    raw = _chat_completion(
        messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = _parse_json_from_llm(raw)

    if not isinstance(data, dict):
        raise AIServiceError(
            "Invalid flashcards response: expected a JSON object."
        )

    cards = data.get("cards")

    if not isinstance(cards, list):
        raise AIServiceError(
            "Invalid flashcards response: 'cards' must be a list."
        )

    cleaned_cards: List[Dict[str, str]] = []

    for item in cards:
        if not isinstance(item, dict):
            continue

        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()

        if not front or not back:
            continue

        cleaned_cards.append(
            {
                "front": front,
                "back": back,
            }
        )

    if not cleaned_cards:
        raise AIServiceError(
            "The LLM returned no valid flashcards."
        )

    return {
        "cards": cleaned_cards[:count]
    }

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
