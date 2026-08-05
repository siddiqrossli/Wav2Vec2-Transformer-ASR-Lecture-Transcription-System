from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import WebConfig


def _library_dir() -> Path:
    p = Path(WebConfig.TRANSCRIPT_LIBRARY_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _transcript_path(job_id: str) -> Path:
    return _library_dir() / f"{job_id}.json"


def _notes_path(job_id: str) -> Path:
    return _library_dir() / f"{job_id}_notes.json"


def save_transcript(job_id: str, transcript: Dict[str, Any]) -> None:
    payload = dict(transcript)
    payload["job_id"] = job_id
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    _transcript_path(job_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_transcript(job_id: str) -> Optional[Dict[str, Any]]:
    path = _transcript_path(job_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_transcripts() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(_library_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name.endswith("_notes.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "job_id": data.get("job_id") or path.stem,
                    "title": data.get("title") or "Untitled",
                    "course_code": data.get("course_code"),
                    "speaker": data.get("speaker"),
                    "duration": data.get("duration"),
                    "processed_at": data.get("processed_at"),
                    "word_count": data.get("word_count"),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def load_notes(job_id: str) -> Dict[str, str]:
    path = _notes_path(job_id)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    notes = data.get("notes") if isinstance(data, dict) else {}
    return notes if isinstance(notes, dict) else {}


def save_notes(job_id: str, notes: Dict[str, str]) -> None:
    payload = {"job_id": job_id, "notes": notes, "updated_at": datetime.now().isoformat(timespec="seconds")}
    _notes_path(job_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_transcript(job_id: str) -> bool:
    jid = str(job_id or "").strip()
    if not jid:
        return False

    transcript_path = _transcript_path(jid)
    notes_path = _notes_path(jid)

    deleted_any = False
    if transcript_path.exists():
        transcript_path.unlink()
        deleted_any = True

    if notes_path.exists():
        notes_path.unlink()

    return deleted_any


def delete_transcripts(job_ids: List[str]) -> Dict[str, Any]:
    deleted = 0
    missing: List[str] = []
    failed: List[str] = []

    for raw_id in job_ids or []:
        jid = str(raw_id or "").strip()
        if not jid:
            continue

        try:
            if delete_transcript(jid):
                deleted += 1
            else:
                missing.append(jid)
        except OSError:
            failed.append(jid)

    return {
        "deleted": deleted,
        "missing": missing,
        "failed": failed,
    }
