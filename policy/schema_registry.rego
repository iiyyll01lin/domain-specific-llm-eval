# OPA policy: Event schema registry integrity
# Ensures every event in the schema_registry.json has required fields
# and that schema versions match ADR-006 semver convention.
#
# Usage:
#   opa eval -d policy/ -i events/schema_registry.json "data.schema_registry.deny"

package schema_registry

import future.keywords.in

required_event_fields := {"name", "version", "schema_file", "sha256"}

deny contains msg if {
    event := input.events[_]
    field := required_event_fields[_]
    not field in object.keys(event)
    msg := sprintf("Event '%v' is missing required field '%v'", [event, field])
}

deny contains msg if {
    event := input.events[_]
    event.sha256 == "TBD"
    msg := sprintf("Event '%v' has sha256 == 'TBD' — run validate_event_schemas.py to populate hashes", [event.name])
}

deny contains msg if {
    event := input.events[_]
    not regex.match(`^\d+\.\d+\.\d+$`, event.version)
    msg := sprintf("Event '%v' version '%v' is not valid semver (expected X.Y.Z)", [event.name, event.version])
}

deny contains msg if {
    # Check for duplicate event name + version combinations
    event_a := input.events[i]
    event_b := input.events[j]
    i < j
    event_a.name == event_b.name
    event_a.version == event_b.version
    msg := sprintf("Duplicate event '%v' version '%v' at indices %v and %v", [event_a.name, event_a.version, i, j])
}
