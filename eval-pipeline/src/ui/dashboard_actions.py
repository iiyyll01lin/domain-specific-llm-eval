from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional


def build_pipeline_command(
    docs: int,
    samples: int,
    config_path: Optional[str] = None,
) -> List[str]:
    command = [
        sys.executable,
        "run_pure_ragas_pipeline.py",
        "--docs",
        str(int(docs)),
        "--samples",
        str(int(samples)),
    ]
    if config_path:
        command.extend(["--config", str(Path(config_path))])
    return command