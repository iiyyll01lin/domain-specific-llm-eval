from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.services.reviewer_api import create_reviewer_service_app

app = create_reviewer_service_app(BASE_DIR)


if __name__ == "__main__":
    uvicorn.run(
        "reviewer_service_api:app",
        host=os.environ.get("REVIEWER_SERVICE_HOST", "0.0.0.0"),
        port=int(os.environ.get("REVIEWER_SERVICE_PORT", "8010")),
        reload=False,
    )