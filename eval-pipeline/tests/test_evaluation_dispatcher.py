"""
Unit tests for TestsetSpec and EvaluationDispatcher.

Written TDD-first before the implementation file is created.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# TestsetSpec dataclass tests
# ---------------------------------------------------------------------------

class TestPathSpec:
    def test_accepts_string_path(self):
        from src.evaluation.evaluation_dispatcher import PathSpec
        spec = PathSpec(path="/tmp/testset.csv")
        assert spec.path == "/tmp/testset.csv"

    def test_accepts_path_object(self, tmp_path: Path):
        from src.evaluation.evaluation_dispatcher import PathSpec
        p = tmp_path / "ts.xlsx"
        spec = PathSpec(path=p)
        assert spec.path == p

    def test_is_frozen(self):
        from src.evaluation.evaluation_dispatcher import PathSpec
        spec = PathSpec(path="/tmp/ts.csv")
        with pytest.raises((AttributeError, TypeError)):
            spec.path = "/other"  # type: ignore[misc]

    def test_equality_by_value(self):
        from src.evaluation.evaluation_dispatcher import PathSpec
        assert PathSpec(path="a") == PathSpec(path="a")
        assert PathSpec(path="a") != PathSpec(path="b")


class TestInMemorySpec:
    def test_stores_data_dict(self):
        from src.evaluation.evaluation_dispatcher import InMemorySpec
        data = {"qa_pairs": [{"user_input": "Q?", "reference": "A."}]}
        spec = InMemorySpec(data=data)
        assert spec.data["qa_pairs"][0]["user_input"] == "Q?"

    def test_is_frozen(self):
        from src.evaluation.evaluation_dispatcher import InMemorySpec
        spec = InMemorySpec(data={})
        with pytest.raises((AttributeError, TypeError)):
            spec.data = {"new": 1}  # type: ignore[misc]


class TestComprehensiveSpec:
    def test_stores_two_paths(self, tmp_path: Path):
        from src.evaluation.evaluation_dispatcher import ComprehensiveSpec
        spec = ComprehensiveSpec(testset_file=tmp_path / "ts.csv", output_dir=tmp_path)
        assert spec.testset_file.name == "ts.csv"
        assert spec.output_dir == tmp_path

    def test_is_frozen(self, tmp_path: Path):
        from src.evaluation.evaluation_dispatcher import ComprehensiveSpec
        spec = ComprehensiveSpec(testset_file=tmp_path / "ts.csv", output_dir=tmp_path)
        with pytest.raises((AttributeError, TypeError)):
            spec.output_dir = Path("/other")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvaluationDispatcher routing tests
# ---------------------------------------------------------------------------

def _make_rag_evaluator() -> MagicMock:
    evaluator = MagicMock()
    evaluator.evaluate_single_testset.return_value = {
        "success": True,
        "result_source": "rag_evaluator",
        "testset_path": "/fake/path.csv",
    }
    evaluator.evaluate_testset.return_value = [
        {"rag_answer": "mock answer", "user_input": "Q?"}
    ]
    evaluator.evaluate_testsets.return_value = {
        "success": True,
        "result_source": "rag_evaluator_batch",
        "total_queries": 3,
    }
    return evaluator


def _make_comprehensive_evaluator() -> MagicMock:
    evaluator = MagicMock()
    evaluator.evaluate_testset.return_value = {
        "success": True,
        "result_source": "comprehensive_rag_evaluator_fixed",
    }
    return evaluator


class TestEvaluationDispatcherRouting:
    def test_path_spec_routes_to_evaluate_single_testset(self):
        from src.evaluation.evaluation_dispatcher import (
            EvaluationDispatcher, PathSpec,
        )
        rag = _make_rag_evaluator()
        dispatcher = EvaluationDispatcher(rag_evaluator=rag)

        spec = PathSpec(path="/tmp/ts.csv")
        result = dispatcher.dispatch(spec)

        rag.evaluate_single_testset.assert_called_once_with("/tmp/ts.csv")
        assert result["success"] is True

    def test_path_spec_with_path_object_normalised_to_str(self, tmp_path: Path):
        from src.evaluation.evaluation_dispatcher import (
            EvaluationDispatcher, PathSpec,
        )
        rag = _make_rag_evaluator()
        dispatcher = EvaluationDispatcher(rag_evaluator=rag)

        p = tmp_path / "ts.csv"
        spec = PathSpec(path=p)
        dispatcher.dispatch(spec)

        rag.evaluate_single_testset.assert_called_once_with(str(p))

    def test_in_memory_spec_routes_to_evaluate_testset(self):
        from src.evaluation.evaluation_dispatcher import (
            EvaluationDispatcher, InMemorySpec,
        )
        rag = _make_rag_evaluator()
        dispatcher = EvaluationDispatcher(rag_evaluator=rag)

        data = {"qa_pairs": [{"user_input": "Q?"}]}
        spec = InMemorySpec(data=data)
        result = dispatcher.dispatch(spec)

        rag.evaluate_testset.assert_called_once_with(data)
        assert isinstance(result, list)

    def test_comprehensive_spec_routes_to_comprehensive_evaluator(self, tmp_path: Path):
        from src.evaluation.evaluation_dispatcher import (
            ComprehensiveSpec, EvaluationDispatcher,
        )
        rag = _make_rag_evaluator()
        comp = _make_comprehensive_evaluator()
        dispatcher = EvaluationDispatcher(rag_evaluator=rag, comprehensive_evaluator=comp)

        spec = ComprehensiveSpec(testset_file=tmp_path / "ts.csv", output_dir=tmp_path)
        result = dispatcher.dispatch(spec)

        comp.evaluate_testset.assert_called_once_with(tmp_path / "ts.csv", tmp_path)
        assert result["success"] is True

    def test_comprehensive_spec_without_evaluator_raises(self, tmp_path: Path):
        from src.evaluation.evaluation_dispatcher import (
            ComprehensiveSpec, EvaluationDispatcher,
        )
        rag = _make_rag_evaluator()
        dispatcher = EvaluationDispatcher(rag_evaluator=rag)  # no comprehensive_evaluator

        spec = ComprehensiveSpec(testset_file=tmp_path / "ts.csv", output_dir=tmp_path)
        with pytest.raises(ValueError, match="ComprehensiveSpec"):
            dispatcher.dispatch(spec)

    def test_dispatch_unknown_spec_type_raises(self):
        from src.evaluation.evaluation_dispatcher import EvaluationDispatcher
        rag = _make_rag_evaluator()
        dispatcher = EvaluationDispatcher(rag_evaluator=rag)

        with pytest.raises(TypeError, match="Unrecognised TestsetSpec"):
            dispatcher.dispatch("not-a-spec")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Backward-compat shim tests
# ---------------------------------------------------------------------------

class TestEvaluationDispatcherShims:
    def test_evaluate_testsets_shim_delegates(self):
        from src.evaluation.evaluation_dispatcher import EvaluationDispatcher
        rag = _make_rag_evaluator()
        dispatcher = EvaluationDispatcher(rag_evaluator=rag)

        result = dispatcher.evaluate_testsets(["/tmp/a.csv", "/tmp/b.csv"])

        rag.evaluate_testsets.assert_called_once_with(["/tmp/a.csv", "/tmp/b.csv"])
        assert result["total_queries"] == 3

    def test_evaluate_testset_shim_delegates(self):
        from src.evaluation.evaluation_dispatcher import EvaluationDispatcher
        rag = _make_rag_evaluator()
        dispatcher = EvaluationDispatcher(rag_evaluator=rag)

        data = {"qa_pairs": [{"user_input": "Q?"}]}
        result = dispatcher.evaluate_testset(data)

        rag.evaluate_testset.assert_called_once_with(data)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestsetSpec Union type annotation smoke test
# ---------------------------------------------------------------------------

def test_testset_spec_union_covers_all_variants():
    """Importing TestsetSpec should expose all three spec types."""
    from src.evaluation.evaluation_dispatcher import (
        ComprehensiveSpec, InMemorySpec, PathSpec, TestsetSpec,
    )
    import typing
    # get_args works on Union
    args = typing.get_args(TestsetSpec)
    assert PathSpec in args
    assert InMemorySpec in args
    assert ComprehensiveSpec in args
