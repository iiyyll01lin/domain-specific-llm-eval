"""MockManufacturingDB — internal SQLite-backed research tool.

Builds an in-process research corpus from the ``sample_documents/`` text files
at the workspace root.  The corpus is indexed once at first query and cached
for the lifetime of the process.

The Researcher agent calls ``MockManufacturingDB().query(...)`` to retrieve
supporting context for each KnowledgeGap without touching the open internet.

Search strategy
---------------
* keyword tokenisation (lowercase, alphanumeric + CJK)
* TF-like scoring: score = number of matching tokens / total query tokens
* Returns the top-K passages (default K=3) with confidence proportional to score
* Each passage is split at sentence boundaries (``。``, ``.\n``, or ``\n\n``)
  to avoid returning wall-of-text chunks

Supported platforms: Linux / macOS (relies only on ``pathlib`` + ``sqlite3``).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corpus paths
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]  # .../domain-specific-llm-eval
_SAMPLE_DOCS_DIR = _WORKSPACE_ROOT / "sample_documents"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[a-z0-9_\u4e00-\u9fff]+")
_SENTENCE_SPLITTER = re.compile(r"(?<=[。\n])|\.\s+|\n{2,}")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _split_passages(text: str, min_len: int = 60, max_len: int = 500) -> List[str]:
    """Split *text* into sentence-bounded passages within length bounds."""
    parts = _SENTENCE_SPLITTER.split(text)
    passages: List[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) + len(part) < max_len:
            buf = (buf + " " + part).strip() if buf else part
        else:
            if buf and len(buf) >= min_len:
                passages.append(buf)
            buf = part
    if buf and len(buf) >= min_len:
        passages.append(buf)
    return passages or [text[:max_len]]


def _extract_keyphrases(text: str, n: int = 5) -> List[str]:
    """Heuristic keyphrase extraction — top-n bigrams by frequency."""
    tokens = _tokenize(text)
    if len(tokens) < 2:
        return tokens[:n]
    bigrams: Dict[str, int] = {}
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i+1]}"
        bigrams[bg] = bigrams.get(bg, 0) + 1
    top = sorted(bigrams, key=lambda k: -bigrams[k])
    return top[:n] if top else tokens[:n]


def _extract_entities(text: str, n: int = 8) -> List[str]:
    """Heuristic entity extraction — capitalised words or CJK clusters."""
    # CJK clusters (2–6 characters) treated as named entities
    cjk = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    # Capitalised English words (likely proper nouns / technical terms)
    caps = re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text)
    seen: set = set()
    result: List[str] = []
    for tok in (cjk + caps):
        if tok not in seen:
            seen.add(tok)
            result.append(tok)
        if len(result) >= n:
            break
    return result


# ---------------------------------------------------------------------------
# In-process corpus cache
# ---------------------------------------------------------------------------

_CORPUS_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_corpus() -> List[Dict[str, Any]]:
    """Load corpus from sample_documents/ once; return cached list afterwards."""
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None:
        return _CORPUS_CACHE

    corpus: List[Dict[str, Any]] = []
    if not _SAMPLE_DOCS_DIR.exists():
        logger.warning("sample_documents/ not found at %s — mock DB will be empty", _SAMPLE_DOCS_DIR)
        _CORPUS_CACHE = corpus
        return corpus

    for txt_file in sorted(_SAMPLE_DOCS_DIR.glob("*.txt")):
        try:
            raw = txt_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s: %s", txt_file, exc)
            continue
        for passage in _split_passages(raw):
            tokens = set(_tokenize(passage))
            corpus.append(
                {
                    "content": passage,
                    "source_uri": f"internal://sample_documents/{txt_file.name}",
                    "tokens": tokens,
                    "keyphrases": _extract_keyphrases(passage),
                    "entities": _extract_entities(passage),
                }
            )

    logger.info("MockManufacturingDB: loaded %d passages from sample_documents/", len(corpus))
    _CORPUS_CACHE = corpus
    return corpus


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class MockManufacturingDB:
    """Simple TF-scored in-process search over the sample_documents corpus.

    This intentionally has no external network calls and is fully deterministic,
    fulfilling the Phase 2 requirement of restricting the Researcher to an
    internal source only.
    """

    def query(
        self,
        keywords: str,
        entity_type: str = "",  # not filtered, kept for interface compatibility
        top_k: int = 3,
        min_confidence: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Search the corpus for *keywords* and return the best matches.

        Parameters
        ----------
        keywords:
            Space- or comma-separated search terms (entity name, gap description, etc.)
        entity_type:
            Unused (reserved for future typed queries).
        top_k:
            Maximum number of results to return.
        min_confidence:
            Minimum token-overlap score to include a passage.

        Returns
        -------
        List[Dict]
            Each dict has keys: ``content``, ``source_uri``, ``confidence``,
            ``supporting_entities``, ``keyphrases``.
        """
        corpus = _load_corpus()
        if not corpus:
            return []

        query_tokens = set(_tokenize(keywords))
        if not query_tokens:
            return []

        scored: List[tuple[float, Dict[str, Any]]] = []
        for entry in corpus:
            doc_tokens: set = entry["tokens"]
            overlap = len(query_tokens & doc_tokens)
            if overlap == 0:
                continue
            # Jaccard-like score: intersection / union
            score = overlap / len(query_tokens | doc_tokens)
            scored.append((score, entry))

        scored.sort(key=lambda t: -t[0])
        results: List[Dict[str, Any]] = []
        for score, entry in scored[:top_k]:
            if score < min_confidence:
                break
            results.append(
                {
                    "content": entry["content"],
                    "source_uri": entry["source_uri"],
                    "confidence": round(score, 4),
                    "supporting_entities": entry["entities"],
                    "keyphrases": entry["keyphrases"],
                }
            )
        return results
