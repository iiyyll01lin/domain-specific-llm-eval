package naming

test_valid_event_key {
  deny with input as {"event_key": "ui.kg.graph.render"} == []
}

test_valid_metric_name {
  deny with input as {"metric_name": "svc_processing_chunk_duration_seconds"} == []
}

test_invalid_event_key_prefix {
  messages := deny with input as {"event_key": "bad.kg.graph.render"}
  count(messages) > 0
}

test_invalid_metric_name_prefix {
  messages := deny with input as {"metric_name": "bad_processing_chunk_duration_seconds"}
  count(messages) > 0
}