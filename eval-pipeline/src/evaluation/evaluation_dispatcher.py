"""
EvaluationDispatcher — Facade for unified testset evaluation routing.

This module defines:
  - ``PathSpec``         — evaluate from a file path
  - ``InMemorySpec``     — evaluate from an in-memory qa_pairs dict
  - ``ComprehensiveSpec``— evaluate via ComprehensiveRAGEvaluatorFixed (file + output_dir)
  - ``TestsetSpec``      — Union of the three spec types
  - ``EvaluationDispatcher`` — routes a ``TestsetSpec`` to the correct backend
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec dataclasses — strongly-typed input variants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathSpec:
    """Dispatch to ``RAGEvaluator.evaluate_single_testset(path)``."""
    path: Union[str, Path]


@dataclass(frozen=True)
class InMemorySpec:
    """Dispatch to ``RAGEvaluator.evaluate_testset(data)``."""
    data: Dict[str, Any]


@dataclass(frozen=True)
class ComprehensiveSpec:
    """Dispatch to ``ComprehensiveRAGEvaluatorFixed.evaluate_testset(file, dir)``."""
    testset_file: Path
    output_dir: Path


#: Union alias — all accepted spec types.
TestsetSpec = Union[PathSpec, InMemorySpec, ComprehensiveSpec]


# ---------------------------------------------------------------------------
# EvaluationDispatcher Facade
# ---------------------------------------------------------------------------

class EvaluationDispatcher:
    """Route a :class:`TestsetSpec` to the correct evaluator backend.

    This is the single public entry-point for all testset evaluation calls.
    The legacy methods on the underlying evaluators remain untouched and
    accessible directly — this facade simply eliminates the need for
    call-site type-guards and provides a unified error-handling path.

    Parameters
    ----------
    rag_evaluator:
        A :class:`~evaluation.rag_evaluator.RAGEvaluator` instance.  Must
        expose ``evaluate_single_testset``, ``evaluate_testset``, and
        ``evaluate_testsets``.
    comprehensive_evaluator:
        Optional :class:`~evaluation.comprehensive_rag_evaluator_fixed.ComprehensiveRAGEvaluatorFixed`
        instance.  Required only when :class:`ComprehensiveSpec` is dispatched.
    """

    def __init__(
        self,
        rag_evaluator: Any,
        comprehensive_evaluator: Optional[Any] = None,
    ) -> None:
        self._rag_evaluator = rag_evaluator
        self._comprehensive_evaluator = comprehensive_evaluator

    # ------------------------------------------------------------------
    # Primary dispatch API
    # ------------------------------------------------------------------

    def dispatch(
        self, spec: TestsetSpec
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Evaluate *spec* using the appropriate backend.

        Parameters
        ----------
        spec:
            A :class:`PathSpec`, :class:`InMemorySpec`, or
            :class:`ComprehensiveSpec` instance.

        Returns
        -------
        Dict or List[Dict]
            The raw return value of the underlying evaluator method, unchanged.

        Raises
        ------
        TypeError
            If *spec* is not a recognised :data:`TestsetSpec` variant.
        ValueError
            If a :class:`ComprehensiveSpec` is given but no
            ``comprehensive_evaluator`` was supplied at construction time.
        """
        if isinstance(spec, PathSpec):
            logger.debug("EvaluationDispatcher → evaluate_single_testset(%s)", spec.path)
            return self._rag_evaluator.evaluate_single_testset(str(spec.path))

        if isinstance(spec, InMemorySpec):
            logger.debug("EvaluationDispatcher → evaluate_testset(in-memory dict)")
            return self._rag_evaluator.evaluate_testset(spec.data)

        if isinstance(spec, ComprehensiveSpec):
            if self._comprehensive_evaluator is None:
                raise ValueError(
                    "A ComprehensiveSpec was dispatched but no comprehensive_evaluator "
                    "was provided to EvaluationDispatcher. Pass the evaluator at "
                    "construction time: EvaluationDispatcher(..., "
                    "comprehensive_evaluator=<ComprehensiveRAGEvaluatorFixed>)."
                )
            logger.debug(
                "EvaluationDispatcher → ComprehensiveRAGEvaluatorFixed.evaluate_testset(%s)",
                spec.testset_file,
            )
            return self._comprehensive_evaluator.evaluate_testset(
                spec.testset_file, spec.output_dir
            )

        raise TypeError(
            f"Unrecognised TestsetSpec type: {type(spec).__name__!r}. "
            "Expected one of: PathSpec, InMemorySpec, ComprehensiveSpec."
        )

    # ------------------------------------------------------------------
    # Backward-compatibility shims (delegate directly to rag_evaluator)
    # ------------------------------------------------------------------

    def evaluate_testsets(self, testset_files: List[str]) -> Dict[str, Any]:
        """Thin shim — delegates to :meth:`RAGEvaluator.evaluate_testsets`.

        Retained so that existing callers (e.g. orchestrator) can switch
        attribute names without touching call sites.
        """
        return self._rag_evaluator.evaluate_testsets(testset_files)

    def evaluate_testset(
        self, testset_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Thin shim — delegates to :meth:`RAGEvaluator.evaluate_testset`."""
        return self._rag_evaluator.evaluate_testset(testset_data)
