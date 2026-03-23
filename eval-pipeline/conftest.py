from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


collect_ignore_glob = [
    "complete_bug_fix_test.py",
    "correct_api_fix_test.py",
    "test_*.py",  # root-level scripts retired; all tests live in tests/
]


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