from __future__ import annotations

import os
import asyncio
import inspect
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PATH_ENTRIES = [
    REPO_ROOT,
    REPO_ROOT / "eval-pipeline",
    REPO_ROOT / "eval-pipeline" / "src",
    REPO_ROOT / "ragas" / "ragas" / "src",
]

for entry in PATH_ENTRIES:
    entry_str = str(entry)
    if entry_str in sys.path:
        sys.path.remove(entry_str)

for entry in reversed(PATH_ENTRIES):
    sys.path.insert(0, str(entry))


_DEFAULT_ENV = {
    "OBJECT_STORE_ENDPOINT": "http://localhost:9000",
    "OBJECT_STORE_REGION": "us-east-1",
    "OBJECT_STORE_ACCESS_KEY": "test-access-key",
    "OBJECT_STORE_SECRET_KEY": "test-secret-key",
    "OBJECT_STORE_BUCKET": "test-bucket",
    "OBJECT_STORE_USE_SSL": "false",
    # Prevent all HuggingFace model downloads during tests so SentenceTransformer
    # and transformers raise immediately instead of hanging on network I/O.
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

for key, value in _DEFAULT_ENV.items():
    os.environ.setdefault(key, value)


collect_ignore_glob = [
    "ragas/ragas/tests/**",
]


from global_tiktoken_patch import apply_global_tiktoken_patch

apply_global_tiktoken_patch()


def pytest_pyfunc_call(pyfuncitem):
    test_function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_function):
        return None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_function(**pyfuncitem.funcargs))
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    return True