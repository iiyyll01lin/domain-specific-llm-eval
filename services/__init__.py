from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_local_ragas() -> None:
	repo_root = Path(__file__).resolve().parent.parent
	ragas_src = repo_root / "ragas" / "ragas" / "src"
	if ragas_src.exists():
		ragas_src_str = str(ragas_src)
		if ragas_src_str not in sys.path:
			sys.path.insert(0, ragas_src_str)


_bootstrap_local_ragas()
