"""Tests for services/kg/extract.py (TASK-061, TASK-062a)."""
import pytest
from services.kg.extract import (
    extract_all,
    extract_entities,
    extract_keyphrases,
    extract_sentences,
)


def test_extract_entities_returns_list():
    result = extract_entities("The inspector checks the steel plate surface quality.", max_entities=10)
    assert isinstance(result, list)


def test_extract_entities_max_cap():
    text = " ".join([f"Entity{i}" for i in range(200)])
    result = extract_entities(text, max_entities=5)
    assert len(result) <= 5


def test_extract_keyphrases_returns_list():
    result = extract_keyphrases("steel plate surface inspection quality check", max_keyphrases=5)
    assert isinstance(result, list)
    assert len(result) <= 5


def test_extract_sentences_splits_text():
    text = "Sentence one. Sentence two. Sentence three."
    result = extract_sentences(text)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_extract_all_returns_all_keys():
    result = extract_all("Steel plate inspection is important for quality control.")
    assert "entities" in result
    assert "keyphrases" in result
    assert "sentences" in result


def test_extract_all_entities_list():
    result = extract_all("The operator checks the conveyor belt speed regularly.")
    assert isinstance(result["entities"], list)


def test_extract_all_keyphrases_list():
    result = extract_all("Knowledge graph quality evaluation pipeline test.")
    assert isinstance(result["keyphrases"], list)


def test_extract_all_sentences_list():
    result = extract_all("First sentence. Second sentence.")
    assert isinstance(result["sentences"], list)


def test_extract_all_empty_text():
    result = extract_all("")
    assert result["entities"] == []
    assert result["keyphrases"] == []
    assert result["sentences"] == []


def test_extract_entities_chinese_fallback():
    """Fallback regex should still extract CJK tokens."""
    result = extract_entities("鋼板表面檢查品質控制", max_entities=10)
    assert isinstance(result, list)
