import pytest

from src.data.dna_sequence_embedder import DNASequenceEmbedder
from src.evaluation.telepathic_intent_evaluator import \
    TelepathicIntentAlignment
from src.evaluation.temporal_causality_evaluator import \
    TemporalCausalityEvaluator
from src.interfaces.swarm_telemetry_ingestor import SwarmTelemetryIngestor


def test_dna_embedder():
    embedder = DNASequenceEmbedder()
    vec = [0.5, -0.2]
    dna = embedder.encode_vector(vec)
    assert len(dna) == 8 # 4 bases per float
    assert all(base in ['A', 'C', 'G', 'T'] for base in dna)
    
    vault = {"Entity1": "AAGGCCAA", "Entity2": "TTTTTTTT"}
    nearest = embedder.find_nearest_neighbor("AAGGCCAT", vault)
    assert nearest == "Entity1"

def test_temporal_evaluator():
    eval = TemporalCausalityEvaluator()
    timeline = ["market boom", "market crash"]
    timeline = eval.inject_temporal_perturbation(timeline, "alien contact")
    
    assert "alien contact" in timeline
    score = eval.score_prediction(["market crash"], "So inflation drops")
    assert score == 0.95

def test_swarm_ingestor():
    ingestor = SwarmTelemetryIngestor()
    ans = ingestor.connect_ros_node("alpha-1")
    assert ans is True
    
    telem = ingestor.ingest_visual_feed("alpha-1", {"detected_objects": ["Tree", "Car"]})
    assert "Tree" in telem
    assert "Car" in telem

def test_telepathic_evaluator():
    eval = TelepathicIntentAlignment()
    intent = eval.decode_eeg([2.0, 4.0, 1.0])
    assert intent == "URGENT_REQUEST"
    
    score = eval.calculate_alignment(intent, "We must act immediately.")
    assert score == 1.0
    
    score_bad = eval.calculate_alignment(intent, "Take your time.")
    assert score_bad == 0.2
