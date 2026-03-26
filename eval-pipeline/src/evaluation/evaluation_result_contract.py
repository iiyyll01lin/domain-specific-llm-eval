from __future__ import annotations

from typing import Any, Dict, Optional


EVALUATION_RESULT_CONTRACT_VERSION = "1.0"


def attach_result_contract(
    payload: Dict[str, Any],
    *,
    result_source: str,
    success: bool,
    error_stage: Optional[str] = None,
    mock_data: bool = False,
) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized["success"] = bool(success)
    normalized["result_source"] = str(result_source)
    normalized["error_stage"] = error_stage
    normalized["mock_data"] = bool(mock_data)
    normalized["contract_version"] = EVALUATION_RESULT_CONTRACT_VERSION
    return normalized


def evaluation_error_result(
    *,
    result_source: str,
    error_stage: str,
    error: str,
    mock_data: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {"error": str(error)}
    if extra:
        payload.update(extra)
    return attach_result_contract(
        payload,
        result_source=result_source,
        success=False,
        error_stage=error_stage,
        mock_data=mock_data,
    )