from __future__ import annotations

from pipeline.enhanced_trackers import (
    CompositionTracker,
    ParametersTracker,
    PerformanceTracker,
)


def test_enhanced_trackers_initialize_and_record_basic_activity() -> None:
    config = {
        "evaluation": {"performance_tracking": {"enabled": True}},
        "testset_generation": {"composition_elements_tracking": {"enabled": True}},
        "reporting": {"final_parameters_tracking": {"enabled": True}},
    }

    performance = PerformanceTracker(config)
    composition = CompositionTracker(config)
    parameters = ParametersTracker(config)

    timing_id = performance.start_timing("test_component", "operation")
    assert timing_id is not None
    duration = performance.end_timing(timing_id)
    assert duration is not None

    composition.track_synthesizer_usage("single-hop", True, "question")
    parameters.track_config_modification("component", "param", "old", "new", "reason")

    assert composition.synthesizer_usage["single-hop"]["successful_generations"] == 1
    assert composition.synthesizer_usage["single-hop"]["total_attempts"] == 1
    assert parameters.config_modifications[0]["parameter"] == "param"


def test_ragas_synthesizer_imports_are_available() -> None:
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

    candidates = [
        "ragas.testset.synthesizers.multi_hop.abstract.MultiHopAbstractQuerySynthesizer",
        "ragas.testset.synthesizers.multi_hop.specific.MultiHopSpecificQuerySynthesizer",
        "ragas.testset.synthesizers.MultiHopAbstractQuerySynthesizer",
        "ragas.testset.synthesizers.MultiHopSpecificQuerySynthesizer",
    ]

    assert SingleHopSpecificQuerySynthesizer is not None
    successes = 0
    for import_path in candidates:
        try:
            module_path, class_name = import_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            successes += 1
        except Exception:
            continue

    assert successes >= 1