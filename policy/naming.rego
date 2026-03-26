# OPA policy: Event schema naming conventions
# Enforces naming rules defined in ADR-006 and the telemetry taxonomy (ADR-005).
#
# Usage:
#   opa eval -d policy/ -i <event_payload.json> "data.naming.deny"
#
# Input shape:
#   {"event_key": "ui.kg.graph.render", "metric_name": "svc_ingestion_document_bytes_total"}

package naming

import future.keywords.in

# ---------------------------------------------------------------------------
# Event key naming: <layer>.<domain>.[<entity>.]<action>[.<qualifier>]
# Layer must be one of: ui, svc, ws, eval, kg
# All components must be lowercase alphanumeric (dots as separators)
# Minimum 3 parts, maximum 5 parts
# ---------------------------------------------------------------------------

allowed_layers := {"ui", "svc", "ws", "eval", "kg"}

event_key_parts(key) := parts if {
    parts := split(key, ".")
}

deny contains msg if {
    key := input.event_key
    parts := event_key_parts(key)
    not parts[0] in allowed_layers
    msg := sprintf("event_key '%v' has invalid layer prefix '%v'; allowed: %v", [key, parts[0], allowed_layers])
}

deny contains msg if {
    key := input.event_key
    parts := event_key_parts(key)
    count(parts) < 3
    msg := sprintf("event_key '%v' too short — must have at least 3 dot-separated parts (layer.domain.action)", [key])
}

deny contains msg if {
    key := input.event_key
    parts := event_key_parts(key)
    count(parts) > 5
    msg := sprintf("event_key '%v' too long — max 5 dot-separated parts", [key])
}

deny contains msg if {
    key := input.event_key
    not regex.match(`^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*){2,4}$`, key)
    msg := sprintf("event_key '%v' must match pattern ^[a-z][a-z0-9]*(\\.[a-z][a-z0-9]*){2,4}$", [key])
}

# ---------------------------------------------------------------------------
# Metric naming: <layer>_<domain>_<entity>_<metric>[_unit]
# All components must be lowercase alphanumeric with underscore separators
# ---------------------------------------------------------------------------

deny contains msg if {
    name := input.metric_name
    parts := split(name, "_")
    not parts[0] in allowed_layers
    msg := sprintf("metric_name '%v' has invalid layer prefix '%v'; allowed: %v", [name, parts[0], allowed_layers])
}

deny contains msg if {
    name := input.metric_name
    count(split(name, "_")) < 4
    msg := sprintf("metric_name '%v' too short — must have at least 4 underscore-separated parts (layer_domain_entity_metric)", [name])
}

deny contains msg if {
    name := input.metric_name
    not regex.match(`^[a-z][a-z0-9]*(_[a-z][a-z0-9]*){3,5}$`, name)
    msg := sprintf("metric_name '%v' must match pattern ^[a-z][a-z0-9]*(_[a-z][a-z0-9]*){3,5}$", [name])
}
