"""
Lightweight TF-IDF search over saved lecture transcripts (no extra ML deps).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transcript_store import _library_dir

WORD_RE = re.compile(r"[a-z][a-z0-9']{2,}")
STOP = frozenset(
    "the and for are but not you all can had her was one our out day get has him his how".split()
)


def _tokenize(text: str) -> List[str]:
    return [w for w in WORD_RE.findall((text or "").lower()) if w not in STOP]


def _build_chunks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    job_id = data.get("job_id", "")
    title = data.get("title", "Untitled")
    course = data.get("course_code") or ""
    speaker = data.get("speaker") or ""
    chunks: List[Dict[str, Any]] = []

    for i, seg in enumerate(data.get("segments") or []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        chunks.append(
            {
                "job_id": job_id,
                "title": title,
                "course_code": course,
                "speaker": speaker,
                "chunk_id": f"{job_id}:{i}",
                "timestamp": seg.get("timestamp", ""),
                "timestamp_seconds": seg.get("timestamp_seconds", 0),
                "text": text,
            }
        )

    if not chunks and data.get("full_text"):
        chunks.append(
            {
                "job_id": job_id,
                "title": title,
                "course_code": course,
                "speaker": speaker,
                "chunk_id": f"{job_id}:0",
                "timestamp": "00:00",
                "timestamp_seconds": 0,
                "text": data["full_text"],
            }
        )

    return chunks


class LectureRAGIndex:
    def __init__(self) -> None:
        self._chunks: List[Dict[str, Any]] = []
        self._vectors: List[Counter[str]] = []
        self._idf: Dict[str, float] = {}

    def rebuild(self) -> int:
        self._chunks = []
        lib = _library_dir()
        for path in lib.glob("*.json"):
            if path.name.endswith("_notes.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._chunks.extend(_build_chunks(data))
            except (json.JSONDecodeError, OSError):
                continue

        n = len(self._chunks)
        df: Counter[str] = Counter()
        self._vectors = []

        for ch in self._chunks:
            tf = Counter(_tokenize(ch["text"]))
            self._vectors.append(tf)
            df.update(tf.keys())

        self._idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}
        return n

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        if not self._chunks:
            self.rebuild()

        q_tf = Counter(_tokenize(query))
        if not q_tf:
            return []

        def vec_score(tf: Counter[str]) -> float:
            terms = set(tf) | set(q_tf)
            return sum((tf.get(t, 0) * self._idf.get(t, 0.0)) * (q_tf.get(t, 0) * self._idf.get(t, 0.0)) for t in terms)

        scored: List[Tuple[float, int]] = []
        for idx, tf in enumerate(self._vectors):
            s = vec_score(tf)
            if s > 0:
                scored.append((s, idx))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, idx in scored[:top_k]:
            ch = dict(self._chunks[idx])
            ch["score"] = round(score, 4)
            results.append(ch)
        return results


_INDEX = LectureRAGIndex()


def search_lectures(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    return _INDEX.search(query, top_k=top_k)


def rebuild_index() -> int:
    return _INDEX.rebuild()
