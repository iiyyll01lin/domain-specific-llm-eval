from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set, Tuple


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9_]+", _normalize(text)))


def _overlap_ratio(left: Sequence[str] | Set[str], right: Sequence[str] | Set[str]) -> float:
    left_tokens = set(left)
    right_tokens = set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class DeterministicSymbolicBackend:
    def evaluate(
        self,
        answer: str,
        context: str,
        was_symbolically_proven: bool,
        expected_fact: str = "",
    ) -> Dict[str, Any]:
        normalized_answer = _normalize(answer)
        normalized_context = _normalize(context)
        normalized_fact = _normalize(expected_fact)
        overlap = _overlap_ratio(_tokenize(normalized_answer), _tokenize(normalized_context))
        fact_match = bool(normalized_fact and normalized_fact in normalized_answer)

        if was_symbolically_proven:
            score = 1.0 if fact_match else round(min(overlap * 0.5, 0.5), 4)
        else:
            score = round(min(overlap, 0.85), 4)

        return {
            "score": score,
            "contract": {
                "backend": "deterministic_symbolic",
                "was_symbolically_proven": was_symbolically_proven,
                "expected_fact": normalized_fact,
                "fact_match": fact_match,
                "context_overlap": round(overlap, 4),
            },
        }


class DeterministicSpatialBackend:
    def evaluate(
        self,
        query: str,
        coordinates: Tuple[int, int, int],
        answer: str,
        context: str,
    ) -> Dict[str, Any]:
        normalized_answer = _normalize(answer)
        normalized_context = _normalize(context)
        anchor_terms = [term for term in ["robotic", "drone", "assembly", "room"] if term in normalized_context]
        matched_terms = [term for term in anchor_terms if term in normalized_answer]
        overlap = _overlap_ratio(_tokenize(normalized_answer), _tokenize(normalized_context))

        if len(matched_terms) >= 2:
            score = 1.0
        elif matched_terms or overlap >= 0.3:
            score = 0.7
        else:
            score = 0.0

        return {
            "score": score,
            "contract": {
                "backend": "deterministic_spatial",
                "coordinates": coordinates,
                "anchor_terms": anchor_terms,
                "matched_terms": matched_terms,
                "query": query,
                "context_overlap": round(overlap, 4),
            },
        }


class DeterministicIntentBackend:
    def decode(self, eeg_vector: List[float]) -> Dict[str, Any]:
        total_activation = float(sum(eeg_vector)) if eeg_vector else 0.0
        peak_activation = max((float(value) for value in eeg_vector), default=0.0)
        if total_activation >= 5.0 or peak_activation >= 3.5:
            intent = "URGENT_REQUEST"
        else:
            intent = "GENERAL_INQUIRY"
        return {
            "intent": intent,
            "contract": {
                "backend": "deterministic_intent",
                "total_activation": round(total_activation, 4),
                "peak_activation": round(peak_activation, 4),
            },
        }

    def evaluate(self, eeg_intent: str, llm_response: str) -> Dict[str, Any]:
        normalized_response = _normalize(llm_response)
        urgent_markers = [marker for marker in ["immediately", "urgent", "now", "asap"] if marker in normalized_response]
        if eeg_intent == "URGENT_REQUEST":
            score = 1.0 if urgent_markers else 0.2
        else:
            score = 0.9 if not urgent_markers else 0.2
        return {
            "score": score,
            "contract": {
                "backend": "deterministic_intent",
                "intent": eeg_intent,
                "urgent_markers": urgent_markers,
            },
        }


class DeterministicTemporalBackend:
    def evaluate(self, timeline: List[str], prediction: str) -> Dict[str, Any]:
        normalized_timeline = [_normalize(item) for item in timeline]
        normalized_prediction = _normalize(prediction)
        timeline_text = " ".join(normalized_timeline)
        supports_market_crash_rule = (
            "market crash" in timeline_text and "inflation drops" in normalized_prediction
        )
        overlap = _overlap_ratio(_tokenize(timeline_text), _tokenize(normalized_prediction))

        if supports_market_crash_rule:
            score = 0.95
        elif overlap >= 0.2:
            score = 0.6
        else:
            score = 0.1

        return {
            "score": score,
            "contract": {
                "backend": "deterministic_temporal",
                "timeline_length": len(timeline),
                "market_crash_rule": supports_market_crash_rule,
                "timeline_prediction_overlap": round(overlap, 4),
            },
        }


class DeterministicSwarmBackend:
    def evaluate(self, question: str, initial_answer: str, experts: List[str]) -> Dict[str, Any]:
        answer_lower = _normalize(initial_answer)
        question_lower = _normalize(question)

        verdicts: List[Dict[str, Any]] = []
        for expert in experts:
            if "security" in expert.lower() and any(
                term in answer_lower for term in ["secret", "password", "pii"]
            ):
                verdicts.append(
                    {
                        "expert": expert,
                        "stance": "revise",
                        "confidence": 0.9,
                        "rationale": "Sensitive content detected; redact or cite safer evidence.",
                    }
                )
            elif "legal" in expert.lower() and "policy" in question_lower and "must" not in answer_lower:
                verdicts.append(
                    {
                        "expert": expert,
                        "stance": "revise",
                        "confidence": 0.75,
                        "rationale": "Policy answer lacks explicit normative wording.",
                    }
                )
            else:
                verdicts.append(
                    {
                        "expert": expert,
                        "stance": "approve",
                        "confidence": 0.8,
                        "rationale": "Answer is acceptable for this domain perspective.",
                    }
                )

        approvals = [verdict for verdict in verdicts if verdict["stance"] == "approve"]
        agreement_rate = len(approvals) / len(verdicts) if verdicts else 0.0
        dissent_reasons = [verdict["rationale"] for verdict in verdicts if verdict["stance"] != "approve"]

        if dissent_reasons:
            final_answer = f"{initial_answer} [Swarm revisions suggested: {'; '.join(dissent_reasons)}]"
        else:
            final_answer = f"{initial_answer} [Debated and Approved]"

        return {
            "final_answer": final_answer,
            "agreement_rate": agreement_rate,
            "verdicts": verdicts,
            "dissent_reasons": dissent_reasons,
            "contract": {
                "backend": "deterministic_swarm",
                "expert_count": len(experts),
                "revision_required": bool(dissent_reasons),
                "policy_question": "policy" in question_lower,
                "sensitive_content_detected": any(
                    term in answer_lower for term in ["secret", "password", "pii"]
                ),
            },
        }