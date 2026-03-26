from src.evaluation.ragas_evaluator import LLMRoleBinding, RagasEvaluator


class _FakeActorLLM:
    def __init__(self, rendered: str):
        self.rendered = rendered
        self.payloads: list[dict[str, object]] = []

    def invoke(self, payload):
        self.payloads.append(payload)
        return self.rendered


def test_prepare_answer_for_evaluation_uses_actor_when_enabled():
    evaluator = object.__new__(RagasEvaluator)
    evaluator.actor_preprocessing_enabled = True
    evaluator.actor_llm = _FakeActorLLM("normalized answer")

    prepared = evaluator._prepare_answer_for_evaluation(
        question="What changed?",
        contexts=["context"],
        answer="draft answer",
        fallback_text="fallback",
    )

    assert prepared == "normalized answer"
    assert evaluator.actor_llm.payloads[0]["question"] == "What changed?"


def test_format_results_includes_actor_critic_role_metadata():
    evaluator = object.__new__(RagasEvaluator)
    evaluator.metric_weights = {"context_precision": 1.0}
    evaluator.llm_roles = {
        "actor": LLMRoleBinding("actor", "http://actor.local", "actor-model", 0.2, 256, True),
        "critic": LLMRoleBinding("critic", "http://critic.local", "critic-model", 0.0, 512, True),
    }

    formatted = evaluator._format_results(results=[], domain_regex_scores=[])

    assert formatted["llm_roles"]["actor"]["model_name"] == "actor-model"
    assert formatted["llm_roles"]["critic"]["model_name"] == "critic-model"