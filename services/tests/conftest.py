import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_ENV = {
    "OBJECT_STORE_ENDPOINT": "http://localhost:9000",
    "OBJECT_STORE_REGION": "us-east-1",
    "OBJECT_STORE_ACCESS_KEY": "test-access-key",
    "OBJECT_STORE_SECRET_KEY": "test-secret-key",
    "OBJECT_STORE_BUCKET": "test-bucket",
    "OBJECT_STORE_USE_SSL": "false",
}

for key, value in _DEFAULT_ENV.items():
    os.environ.setdefault(key, value)
