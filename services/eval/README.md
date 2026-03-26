# Evaluation Service Metrics

The evaluation microservice exports the following Prometheus metrics:

- `evaluation_run_created_total{result="created|duplicate"}` – counts run submissions stored by the guard layer.
- `rag_request_latency_seconds{outcome="success|failure|unknown"}` – histogram tracking RAG invocation latency with buckets tuned for sub-second and multi-second calls.
- `rag_request_attempt_total{outcome="success|failure|unknown"}` – counts the total number of RAG adapter attempts (including retries) grouped by outcome.

Each RAG invocation additionally emits a structured log entry `rag.request_metrics` that carries the active `trace_id`, observed latency, attempt count, and (when available) a provider error code. These logs allow cross-referencing latency spikes with distributed traces during incident response.

## Baseline Metric Plugins

The baseline plugin bundle shipped with the service implements lightweight,
deterministic string-overlap heuristics that provide immediate scoring without
depending on heavyweight LLM calls. The following plugins conform to the
`MetricPlugin` contract:

- `FaithfulnessMetric` – measures how much of the answer text is supported by
	the retrieved contexts by comparing token overlap.
- `AnswerRelevancyMetric` – checks whether the answer addresses the original
	question by comparing shared tokens between the question and answer.
- `ContextPrecisionMetric` – tracks how much of the retrieved context content is
	actually referenced in the answer to highlight unused evidence.

All metrics expose matched/total token counts in their metadata payloads to aid
downstream aggregation.
