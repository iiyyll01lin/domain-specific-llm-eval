from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, TypedDict


class MultimodalMetric(TypedDict):
    mean: float
    scores: List[float]
    samples_evaluated: int


class MultimodalEvaluationResult(TypedDict):
    metrics: Dict[str, MultimodalMetric]
    modalities_present: Dict[str, int]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_\-]+", text.lower())


class MultimodalResponseEvaluator:
    """Heuristic multimodal scorer used when answers contain image/audio/text context."""

    def _normalize_contexts(self, contexts: Iterable[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for context in contexts:
            if isinstance(context, dict):
                entry = dict(context)
                entry.setdefault("type", entry.get("modality", "text"))
                entry.setdefault("content", "")
                normalized.append(entry)
            else:
                normalized.append({"type": "text", "content": str(context)})
        return normalized

    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: Iterable[Any],
    ) -> Dict[str, float]:
        normalized_contexts = self._normalize_contexts(contexts)
        answer_tokens = set(_tokenize(answer))
        question_tokens = set(_tokenize(question))
        context_tokens = set()
        matched_modalities = 0

        for context in normalized_contexts:
            content_parts = [
                str(context.get("content", "")),
                str(context.get("ocr_text", "")),
                str(context.get("transcript", "")),
                str(context.get("spatial_context", "")),
            ]
            modality_tokens = set(_tokenize(" ".join(content_parts)))
            context_tokens |= modality_tokens
            if answer_tokens & modality_tokens:
                matched_modalities += 1

        overlap_with_context = len(answer_tokens & context_tokens)
        answer_overlap_ratio = (
            overlap_with_context / max(len(answer_tokens), 1)
            if answer_tokens
            else 0.0
        )
        question_alignment = (
            len(question_tokens & context_tokens) / max(len(question_tokens), 1)
            if question_tokens
            else 0.0
        )
        modality_coverage = (
            matched_modalities / max(len(normalized_contexts), 1)
            if normalized_contexts
            else 0.0
        )

        return {
            "multimodal_faithfulness": round(answer_overlap_ratio, 4),
            "multimodal_relevance": round((answer_overlap_ratio + question_alignment) / 2, 4),
            "multimodal_coverage": round(modality_coverage, 4),
        }

    def evaluate_responses(
        self, rag_responses: List[Dict[str, Any]]
    ) -> MultimodalEvaluationResult:
        metrics: Dict[str, List[float]] = {
            "multimodal_faithfulness": [],
            "multimodal_relevance": [],
            "multimodal_coverage": [],
        }
        modalities_present: Counter[str] = Counter()

        for response in rag_responses:
            contexts = response.get("rag_contexts") or response.get("contexts") or []
            if not contexts:
                continue
            normalized_contexts = self._normalize_contexts(contexts)
            for context in normalized_contexts:
                modalities_present[str(context.get("type", "text"))] += 1
            sample_scores = self.evaluate_response(
                question=str(response.get("question") or response.get("user_input") or ""),
                answer=str(response.get("answer") or response.get("rag_answer") or ""),
                contexts=normalized_contexts,
            )
            for metric_name, score in sample_scores.items():
                metrics[metric_name].append(score)

        formatted_metrics: Dict[str, MultimodalMetric] = {}
        for metric_name, scores in metrics.items():
            formatted_metrics[metric_name] = {
                "mean": (sum(scores) / len(scores)) if scores else 0.0,
                "scores": scores,
                "samples_evaluated": len(scores),
            }

        return {
            "metrics": formatted_metrics,
            "modalities_present": dict(modalities_present),
        }