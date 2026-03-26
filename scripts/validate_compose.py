#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT_DIR / "docker-compose.services.yml"
ENV_FILE = ROOT_DIR / ".env.compose"
SERVICES = ["ingestion", "processing", "testset", "eval", "reporting", "adapter", "kg"]


def main() -> int:
    failures = []

    if not COMPOSE_FILE.exists():
        failures.append(f"compose file missing: {COMPOSE_FILE}")
    else:
        cmd = ["docker", "compose", "-f", str(COMPOSE_FILE)]
        if ENV_FILE.exists():
            cmd.extend(["--env-file", str(ENV_FILE)])
        cmd.extend(["config", "--services"])

        try:
            output = subprocess.check_output(cmd, text=True, cwd=ROOT_DIR)
            configured_services = set(output.split())
            for service in SERVICES:
                if service not in configured_services:
                    failures.append(f"service {service} missing in compose")
        except subprocess.CalledProcessError as exc:
            failures.append(f"compose config error: {exc}")

    result = {"checks": len(SERVICES), "failures": failures}
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
