"""Sample metric plugin for demonstration purposes."""

PLUGIN_KIND = "metric"
PLUGIN_NAME = "sample_metric"
PLUGIN_VERSION = "0.1.0"


def register() -> dict:
    """Return plugin metadata payload."""
    return {
        "name": PLUGIN_NAME,
        "kind": PLUGIN_KIND,
        "version": PLUGIN_VERSION,
        "description": "Demonstration metric plugin shipped with the repo.",
    }
