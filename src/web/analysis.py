"""
Lightweight transcript analysis: keyword extraction and extractive summary.
Uses only the standard library (no extra ML dependencies).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List

_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing don down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself just ll m me more most mr
    ms my myself no nor not now of off on once only or other our ours ourselves out
    over own re s same she should so some such t than that the their theirs them
    themselves then there these they this those through to too under until up very
    ve was we were what when where which while who whom why will with would you
    your yours yourself yourselves
    """.split()
)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _content_words(text: str) -> List[str]:
    return [w for w in _tokenize(text) if len(w) > 2 and w not in _STOPWORDS]


def extract_keywords(text: str, top_n: int = 12) -> List[str]:
    """Return the most frequent content words in the transcript."""
    words = _content_words(text)
    if not words:
        return []
    counts = Counter(words)
    return [word for word, _ in counts.most_common(top_n)]


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    sentences = [s.strip() for s in parts if s.strip()]
    if len(sentences) <= 1 and text:
        return [text]
    return sentences


def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """
    Pick top-scoring sentences by summed keyword frequency (extractive summary).
    Preserves original sentence order in the output.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    word_freq = Counter(_content_words(text))
    if not word_freq:
        return sentences[0]

    scored: List[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        score = sum(word_freq.get(w, 0) for w in _tokenize(sentence))
        scored.append((float(score), idx, sentence))

    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:num_sentences]
    top.sort(key=lambda x: x[1])
    return " ".join(s for _, _, s in top)


def word_count(text: str) -> int:
    return len(_tokenize(text))


def reading_time_label(word_total: int, wpm: int = 200) -> str:
    """Human-readable estimated reading time."""
    if word_total <= 0:
        return "0 min"
    minutes = max(1, round(word_total / max(wpm, 1)))
    if minutes == 1:
        return "1 min read"
    return f"{minutes} min read"
