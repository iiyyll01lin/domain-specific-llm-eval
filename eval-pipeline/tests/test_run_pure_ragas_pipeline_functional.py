from types import SimpleNamespace

import run_pure_ragas_pipeline as pipeline


class _FakeSynthesizer:
    def __init__(self, llm, name, available=True):
        self.llm = llm
        self.name = name
        self._available = available

    def get_node_clusters(self, kg):
        return [object()] if self._available else []


def test_build_generation_settings_uses_weighted_distribution(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "SingleHopSpecificQuerySynthesizer",
        lambda llm: _FakeSynthesizer(llm, "single_hop_specific"),
    )
    monkeypatch.setattr(
        pipeline,
        "MultiHopAbstractQuerySynthesizer",
        lambda llm: _FakeSynthesizer(llm, "multi_hop_abstract"),
    )
    monkeypatch.setattr(
        pipeline,
        "MultiHopSpecificQuerySynthesizer",
        lambda llm: _FakeSynthesizer(llm, "multi_hop_specific", available=False),
    )

    config = {
        "testset_generation": {
            "prompt_config": {"profile": "default"},
            "generation": {
                "async_generation": {"enabled": True, "max_workers": 12, "batch_size": 5},
                "query_distribution": [
                    {"synthesizer": "single_hop_specific", "weight": 0.4},
                    {"synthesizer": "multi_hop_abstract", "weight": 0.6},
                    {"synthesizer": "multi_hop_specific", "weight": 0.2},
                ],
            },
        }
    }

    settings = pipeline.build_generation_settings(
        config,
        llm=object(),
        kg=SimpleNamespace(nodes=[object()]),
    )

    assert settings.run_config.max_workers == 12
    assert settings.batch_size == 5
    assert len(settings.personas) >= 2
    assert len(settings.query_distribution) == 2
    assert abs(sum(weight for _, weight in settings.query_distribution) - 1.0) < 1e-9


def test_generate_ragas_testset_passes_run_config_and_batch_size(monkeypatch):
    captured = {}

    class FakeTestsetGenerator:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate(self, **kwargs):
            captured["generate"] = kwargs
            eval_sample = SimpleNamespace(
                user_input="What changed?",
                reference_contexts=["ctx"],
                reference="ans",
            )
            sample = SimpleNamespace(
                eval_sample=eval_sample,
                synthesizer_name="single_hop_specific",
            )
            return SimpleNamespace(samples=[sample])

    monkeypatch.setattr(pipeline, "TestsetGenerator", FakeTestsetGenerator)

    settings = pipeline.GenerationSettings(
        run_config=pipeline.RunConfig(max_workers=9),
        batch_size=4,
        personas=[pipeline.Persona(name="User", role_description="Role")],
        persona_records=[{"name": "User", "role_description": "Role"}],
        query_distribution=[(SimpleNamespace(name="single_hop_specific"), 1.0)],
        query_distribution_records=[{"synthesizer": "single_hop_specific", "weight": 1.0}],
        prompt_profile="default",
        fallback_templates={},
    )

    samples = pipeline.generate_ragas_testset(
        kg=SimpleNamespace(nodes=[]),
        generator_llm=object(),
        generator_embeddings=object(),
        generation_settings=settings,
        num_samples=3,
    )

    assert samples[0]["question"] == "What changed?"
    assert captured["init"]["persona_list"] == settings.personas
    assert captured["generate"]["run_config"].max_workers == 9
    assert captured["generate"]["batch_size"] == 4
