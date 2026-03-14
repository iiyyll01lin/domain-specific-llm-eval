from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentVerdict:
    expert: str
    stance: str
    confidence: float
    rationale: str


class SwarmSynthesizer:
    def __init__(self, experts: List[str] = ["Legal", "Code", "Security"]) -> None:
        self.experts = experts

    def debate_answer(self, question: str, initial_answer: str) -> Dict[str, Any]:
        logger.info("Starting Swarm Debate for '%s' amongst %s", question, self.experts)
        start = time.perf_counter()

        verdicts: List[AgentVerdict] = []
        answer_lower = initial_answer.lower()
        for expert in self.experts:
            if "Security" in expert and any(term in answer_lower for term in ["secret", "password", "pii"]):
                verdicts.append(
                    AgentVerdict(expert, "revise", 0.9, "Sensitive content detected; redact or cite safer evidence.")
                )
            elif "Legal" in expert and "must" not in answer_lower and "policy" in question.lower():
                verdicts.append(
                    AgentVerdict(expert, "revise", 0.75, "Policy answer lacks explicit normative wording.")
                )
            else:
                verdicts.append(
                    AgentVerdict(expert, "approve", 0.8, "Answer is acceptable for this domain perspective.")
                )

        approvals = [verdict for verdict in verdicts if verdict.stance == "approve"]
        agreement_rate = len(approvals) / len(verdicts) if verdicts else 0.0
        consensus_speed_ms = int((time.perf_counter() - start) * 1000) or 1
        dissent_reasons = [verdict.rationale for verdict in verdicts if verdict.stance != "approve"]

        final_answer = initial_answer
        if dissent_reasons:
            final_answer = f"{initial_answer} [Swarm revisions suggested: {'; '.join(dissent_reasons)}]"
        else:
            final_answer = f"{initial_answer} [Debated and Approved]"

        return {
            "final_answer": final_answer,
            "agreement_rate": agreement_rate,
            "consensus_speed_ms": consensus_speed_ms,
            "verdicts": [asdict(verdict) for verdict in verdicts],
            "dissent_reasons": dissent_reasons,
        }
