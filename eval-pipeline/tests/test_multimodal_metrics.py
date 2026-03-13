from src.evaluation.multimodal_metrics import MultimodalResponseEvaluator


def test_multimodal_response_evaluator_scores_context_overlap() -> None:
    evaluator = MultimodalResponseEvaluator()

    result = evaluator.evaluate_responses(
        [
            {
                "question": "What is shown in the panel and audio?",
                "answer": "The robotic arm is described in the transcript.",
                "contexts": [
                    {"type": "image", "content": "robotic arm panel", "ocr_text": "robotic arm"},
                    {"type": "audio", "content": "safety transcript", "transcript": "robotic arm maintenance"},
                ],
            }
        ]
    )

    assert result["metrics"]["multimodal_faithfulness"]["mean"] > 0.3
    assert result["modalities_present"]["image"] == 1
    assert result["modalities_present"]["audio"] == 1