# Evaluation Service Metrics

The evaluation microservice exports the following Prometheus metrics:

- `evaluation_run_created_total{result="created|duplicate"}` – counts run submissions stored by the guard layer.
- `rag_request_latency_seconds{outcome="success|failure|unknown"}` – histogram tracking RAG invocation latency with buckets tuned for sub-second and multi-second calls.
- `rag_request_attempt_total{outcome="success|failure|unknown"}` – counts the total number of RAG adapter attempts (including retries) grouped by outcome.

Each RAG invocation additionally emits a structured log entry `rag.request_metrics` that carries the active `trace_id`, observed latency, attempt count, and (when available) a provider error code. These logs allow cross-referencing latency spikes with distributed traces during incident response.
