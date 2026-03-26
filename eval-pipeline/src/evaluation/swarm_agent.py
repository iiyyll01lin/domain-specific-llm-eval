from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from .deterministic_backends import DeterministicSwarmBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentVerdict:
    expert: str
    stance: str
    confidence: float
    rationale: str


class SwarmSynthesizer:
    def __init__(
        self,
        experts: List[str] = ["Legal", "Code", "Security"],
        backend: DeterministicSwarmBackend | None = None,
    ) -> None:
        self.experts = experts
        self.backend = backend or DeterministicSwarmBackend()

    def debate_answer(self, question: str, initial_answer: str) -> Dict[str, Any]:
        logger.info("Starting Swarm Debate for '%s' amongst %s", question, self.experts)
        start = time.perf_counter()
        result = self.backend.evaluate(question, initial_answer, self.experts)
        result["consensus_speed_ms"] = int((time.perf_counter() - start) * 1000) or 1
        verdicts = [AgentVerdict(**verdict) for verdict in result.get("verdicts", [])]
        result["verdicts"] = [asdict(verdict) for verdict in verdicts]
        return result
