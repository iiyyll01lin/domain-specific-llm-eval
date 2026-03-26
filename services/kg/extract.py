"""Entity and keyphrase extraction with graceful fallbacks.

TASK-061 / TASK-062a: spaCy NER + KeyBERT keyphrases; fallback to regex
segmentation when optional dependencies are absent.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional heavyweight imports — resolved at call time so the module
# can always be imported even without spaCy/KeyBERT installed.
# ---------------------------------------------------------------------------

_NLP = None
_NLP_LOADED = False

_KBMODEL = None
_KB_LOADED = False


def _load_spacy():
    global _NLP, _NLP_LOADED
    if _NLP_LOADED:
        return _NLP
    try:
        import spacy  # noqa: F401

        _NLP = spacy.load("zh_core_web_sm")
    except Exception:
        try:
            import spacy

            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = None
    _NLP_LOADED = True
    return _NLP


def _load_keybert():
    global _KBMODEL, _KB_LOADED
    if _KB_LOADED:
        return _KBMODEL
    try:
        from keybert import KeyBERT  # type: ignore

        _KBMODEL = KeyBERT()
    except Exception:
        _KBMODEL = None
    _KB_LOADED = True
    return _KBMODEL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy if available, else simple regex."""
    nlp = _load_spacy()
    if nlp is not None:
        doc = nlp(text[:10_000])  # cap for performance
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    # Fallback: split on sentence-ending punctuation.
    parts = re.split(r"(?<=[。！？.!?])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def extract_entities(text: str, max_entities: int = 30) -> List[str]:
    """Extract named entities using spaCy NER; fallback to word frequency heuristic."""
    nlp = _load_spacy()
    if nlp is not None:
        doc = nlp(text[:10_000])
        seen: dict[str, int] = {}
        for ent in doc.ents:
            t = ent.text.strip()
            if t:
                seen[t] = seen.get(t, 0) + 1
        # Sort by frequency, return top N
        ranked = sorted(seen, key=lambda k: -seen[k])
        return ranked[:max_entities]
    # Fallback: extract CJK/alpha tokens of length >= 2 as entities
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text)
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq, key=lambda k: -freq[k])
    return ranked[:max_entities]


def extract_keyphrases(text: str, max_keyphrases: int = 10) -> List[str]:
    """Extract keyphrases using KeyBERT; fallback to top frequent bigrams."""
    kb = _load_keybert()
    if kb is not None:
        try:
            results = kb.extract_keywords(
                text[:5_000],
                keyphrase_ngram_range=(1, 3),
                stop_words=None,
                top_n=max_keyphrases,
            )
            return [kw for kw, _score in results]
        except Exception:
            pass
    # Fallback: sliding window bigrams
    tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text)
    bigrams: dict[str, int] = {}
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i+1]}"
        bigrams[bg] = bigrams.get(bg, 0) + 1
    ranked = sorted(bigrams, key=lambda k: -bigrams[k])
    return ranked[:max_keyphrases]


def extract_all(
    text: str,
    max_entities: int = 30,
    max_keyphrases: int = 10,
) -> Dict[str, List[str]]:
    """Return entities, keyphrases, and sentences for a document text."""
    return {
        "entities": extract_entities(text, max_entities=max_entities),
        "keyphrases": extract_keyphrases(text, max_keyphrases=max_keyphrases),
        "sentences": extract_sentences(text),
    }
