package schema_registry

valid_registry := {
  "events": [
    {
      "name": "document.ingested",
      "version": "1.0.0",
      "schema_file": "schemas/document.ingested.v1.json",
      "sha256": "abc123"
    }
  ]
}

invalid_registry := {
  "events": [
    {
      "name": "document.ingested",
      "version": "1.0",
      "schema_file": "schemas/document.ingested.v1.json",
      "sha256": "TBD"
    }
  ]
}

test_valid_schema_registry {
  deny with input as valid_registry == []
}

test_invalid_schema_registry {
  messages := deny with input as invalid_registry
  count(messages) >= 2
}