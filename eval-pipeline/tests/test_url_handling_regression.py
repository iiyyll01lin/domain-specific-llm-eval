from __future__ import annotations

from src.evaluation.ragas_evaluator import RagasEvaluator


def test_normalize_chat_endpoint_removes_v1_and_chat_completions() -> None:
    evaluator = object.__new__(RagasEvaluator)

    assert (
        evaluator._normalize_chat_endpoint("http://llm-proxy.tao.inventec.net")
        == "http://llm-proxy.tao.inventec.net"
    )
    assert (
        evaluator._normalize_chat_endpoint("http://llm-proxy.tao.inventec.net/v1")
        == "http://llm-proxy.tao.inventec.net"
    )
    assert (
        evaluator._normalize_chat_endpoint(
            "http://llm-proxy.tao.inventec.net/v1/chat/completions"
        )
        == "http://llm-proxy.tao.inventec.net"
    )
