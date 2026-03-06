"""Unit tests for scripts/validate_event_schemas.py and scripts/validate_telemetry_taxonomy.py."""
from __future__ import annotations
import json
import pathlib
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VALIDATE_EVENTS_SCRIPT = REPO_ROOT / "scripts" / "validate_event_schemas.py"
VALIDATE_TAXONOMY_SCRIPT = REPO_ROOT / "scripts" / "validate_telemetry_taxonomy.py"


# ---------------------------------------------------------------------------
# validate_event_schemas.py
# ---------------------------------------------------------------------------

class TestValidateEventSchemas:
    def _run(self, *extra_args):
        result = subprocess.run(
            [sys.executable, str(VALIDATE_EVENTS_SCRIPT), *extra_args],
            capture_output=True,
            text=True,
        )
        return result

    def test_passes_on_real_registry(self):
        """Validation script exits 0 against the actual registry."""
        result = self._run()
        assert result.returncode == 0, result.stdout + result.stderr

    def test_all_schemas_valid_message(self):
        """Outputs confirmation message on success."""
        result = self._run()
        assert "valid" in result.stdout.lower()

    def test_missing_registry_exits_nonzero(self, tmp_path, monkeypatch):
        """If registry file does not exist, script exits non-zero."""
        # We invoke a tiny wrapper that monkeypatches the REGISTRY path
        fake_script = tmp_path / "check.py"
        fake_script.write_text(
            f"""
import sys
sys.path.insert(0, '{REPO_ROOT / "scripts"}')
import validate_event_schemas as v
import pathlib
v.REGISTRY = pathlib.Path('/nonexistent/schema_registry.json')
sys.exit(v.main())
"""
        )
        result = subprocess.run([sys.executable, str(fake_script)], capture_output=True, text=True)
        assert result.returncode != 0

    def test_duplicate_event_names_fail(self, tmp_path):
        """Duplicate event name+version pair causes failure."""
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "dup.v1.json"
        schema_file.write_text(json.dumps({"type": "object", "properties": {}}))

        registry = {
            "registry_version": 1,
            "events": [
                {"name": "dup.event", "version": "1.0.0", "schema_file": "schemas/dup.v1.json", "sha256": "TBD"},
                {"name": "dup.event", "version": "1.0.0", "schema_file": "schemas/dup.v1.json", "sha256": "TBD"},
            ],
        }
        registry_file = tmp_path / "schema_registry.json"
        registry_file.write_text(json.dumps(registry))

        fake_script = tmp_path / "check.py"
        fake_script.write_text(
            f"""
import sys
sys.path.insert(0, '{REPO_ROOT / "scripts"}')
import validate_event_schemas as v
import pathlib
v.REGISTRY = pathlib.Path(r'{registry_file}')
sys.exit(v.main())
"""
        )
        result = subprocess.run([sys.executable, str(fake_script)], capture_output=True, text=True)
        assert result.returncode == 1
        assert "Duplicate" in result.stdout

    def test_missing_schema_file_fails(self, tmp_path):
        """If event refers to a missing schema file, validation fails."""
        registry = {
            "registry_version": 1,
            "events": [
                {"name": "missing.event", "version": "1.0.0", "schema_file": "schemas/ghost.v1.json", "sha256": "TBD"},
            ],
        }
        registry_file = tmp_path / "schema_registry.json"
        registry_file.write_text(json.dumps(registry))

        fake_script = tmp_path / "check.py"
        fake_script.write_text(
            f"""
import sys
sys.path.insert(0, '{REPO_ROOT / "scripts"}')
import validate_event_schemas as v
import pathlib
v.REGISTRY = pathlib.Path(r'{registry_file}')
sys.exit(v.main())
"""
        )
        result = subprocess.run([sys.executable, str(fake_script)], capture_output=True, text=True)
        assert result.returncode == 1
        assert "not found" in result.stdout.lower()

    def test_hash_mismatch_fails(self, tmp_path):
        """If stored sha256 differs from actual file hash, validation fails."""
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "ev.v1.json"
        schema_file.write_text(json.dumps({"type": "object", "properties": {}}))

        registry = {
            "registry_version": 1,
            "events": [
                {
                    "name": "ev.event",
                    "version": "1.0.0",
                    "schema_file": "schemas/ev.v1.json",
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                },
            ],
        }
        registry_file = tmp_path / "schema_registry.json"
        registry_file.write_text(json.dumps(registry))

        fake_script = tmp_path / "check.py"
        fake_script.write_text(
            f"""
import sys
sys.path.insert(0, '{REPO_ROOT / "scripts"}')
import validate_event_schemas as v
import pathlib
v.REGISTRY = pathlib.Path(r'{registry_file}')
sys.exit(v.main())
"""
        )
        result = subprocess.run([sys.executable, str(fake_script)], capture_output=True, text=True)
        assert result.returncode == 1
        assert "mismatch" in result.stdout.lower()


# ---------------------------------------------------------------------------
# validate_telemetry_taxonomy.py
# ---------------------------------------------------------------------------

class TestValidateTelemetryTaxonomy:
    def _run(self):
        result = subprocess.run(
            [sys.executable, str(VALIDATE_TAXONOMY_SCRIPT)],
            capture_output=True,
            text=True,
        )
        return result

    def test_passes_on_real_taxonomy(self):
        """Taxonomy validator exits 0 against the actual taxonomy file."""
        result = self._run()
        assert result.returncode == 0, result.stdout + result.stderr

    def test_valid_message_present(self):
        """Outputs confirmation message on success."""
        result = self._run()
        assert "valid" in result.stdout.lower()

    def test_missing_taxonomy_exits_nonzero(self, tmp_path):
        """Missing taxonomy file causes non-zero exit."""
        fake_script = tmp_path / "check.py"
        fake_script.write_text(
            f"""
import sys
sys.path.insert(0, '{REPO_ROOT / "scripts"}')
import validate_telemetry_taxonomy as v
import pathlib
v.TAX = pathlib.Path('/nonexistent/telemetry_taxonomy.json')
sys.exit(v.main())
"""
        )
        result = subprocess.run([sys.executable, str(fake_script)], capture_output=True, text=True)
        assert result.returncode != 0

    def test_duplicate_event_keys_fail(self, tmp_path):
        """Duplicate event keys trigger failure."""
        taxonomy = {
            "taxonomy_version": 1,
            "events": [
                {"key": "ui.kg.render"},
                {"key": "ui.kg.render"},
            ],
            "metrics": [],
            "validation_rules": {
                "event_key_regex": "^.+$",
                "metric_name_regex": "^.+$",
                "no_duplicate_events": True,
                "no_duplicate_metrics": True,
            },
        }
        tax_file = tmp_path / "telemetry_taxonomy.json"
        tax_file.write_text(json.dumps(taxonomy))

        fake_script = tmp_path / "check.py"
        fake_script.write_text(
            f"""
import sys
sys.path.insert(0, '{REPO_ROOT / "scripts"}')
import validate_telemetry_taxonomy as v
import pathlib
v.TAX = pathlib.Path(r'{tax_file}')
sys.exit(v.main())
"""
        )
        result = subprocess.run([sys.executable, str(fake_script)], capture_output=True, text=True)
        assert result.returncode == 1
        assert "Duplicate" in result.stdout

    def test_bad_event_key_regex_fails(self, tmp_path):
        """Event key not matching the naming regex triggers failure."""
        taxonomy = {
            "taxonomy_version": 1,
            "events": [
                {"key": "INVALID_KEY__CAPS"},
            ],
            "metrics": [],
            "validation_rules": {
                "event_key_regex": "^(ui|svc|ws|eval|kg)\\.[a-z0-9]+(\\.[a-z0-9]+){1,4}$",
                "metric_name_regex": "^.+$",
                "no_duplicate_events": True,
                "no_duplicate_metrics": True,
            },
        }
        tax_file = tmp_path / "telemetry_taxonomy.json"
        tax_file.write_text(json.dumps(taxonomy))

        fake_script = tmp_path / "check.py"
        fake_script.write_text(
            f"""
import sys
sys.path.insert(0, '{REPO_ROOT / "scripts"}')
import validate_telemetry_taxonomy as v
import pathlib
v.TAX = pathlib.Path(r'{tax_file}')
sys.exit(v.main())
"""
        )
        result = subprocess.run([sys.executable, str(fake_script)], capture_output=True, text=True)
        assert result.returncode == 1
        assert "regex mismatch" in result.stdout.lower()
