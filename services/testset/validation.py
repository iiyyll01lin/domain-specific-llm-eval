from __future__ import annotations

from typing import Annotated, Any, Iterable, Mapping

from fastapi import status
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from services.common.errors import ServiceError

_ALLOWED_METHODS = {"configurable", "ragas", "hybrid"}
_MAX_TOTAL_SAMPLES = 5000
_MAX_STRATEGIES = 16
_MAX_STRATEGY_LENGTH = 64
_MAX_SEED_VALUE = 2**32 - 1
_ERROR_CODE = "testset_config_invalid"


def _strip_optional_string(value: Any) -> Any:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return value


class PersonaProfile(BaseModel):
    """Optional persona metadata attached to a testset generation request."""

    model_config = ConfigDict(extra="allow")

    id: str | None = Field(default=None, min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    role: str | None = Field(default=None, min_length=1, max_length=64)
    locale: str | None = Field(default=None, min_length=2, max_length=10)
    description: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("id", "name", "role", "locale", "description", mode="before")
    @classmethod
    def _strip_strings(cls, value: Any) -> Any:
        return _strip_optional_string(value)

    @field_validator("locale")
    @classmethod
    def _normalise_locale(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.lower()


PositiveSampleCount = Annotated[int, Field(gt=0, le=_MAX_TOTAL_SAMPLES)]
SeedType = Annotated[int, Field(ge=0, le=_MAX_SEED_VALUE)]


class TestsetConfigModel(BaseModel):
    """Canonical representation of a testset generation request configuration."""

    model_config = ConfigDict(extra="allow")

    method: str
    max_total_samples: PositiveSampleCount = Field(default=50)
    samples_per_document: PositiveSampleCount | None = None
    seed: SeedType | None = None
    selected_strategies: list[str] = Field(default_factory=list)
    persona_profile: PersonaProfile | None = None

    @field_validator("method", mode="before")
    @classmethod
    def _normalise_method(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        if value not in _ALLOWED_METHODS:
            raise ValueError(f"unsupported method '{value}'")
        return value

    @field_validator("selected_strategies", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            candidates: Iterable[str] = [value]
        elif isinstance(value, Iterable):
            candidates = value  # type: ignore[assignment]
        else:
            raise TypeError("selected_strategies must be a list of strings")

        seen: set[str] = set()
        result: list[str] = []
        for item in candidates:
            if not isinstance(item, str):
                raise TypeError("selected_strategies entries must be strings")
            trimmed = item.strip()
            if not trimmed:
                continue
            lowered = trimmed.lower()
            if lowered in seen:
                continue
            if len(trimmed) > _MAX_STRATEGY_LENGTH:
                raise ValueError("selected strategy length exceeds limit")
            result.append(trimmed)
            seen.add(lowered)

        if len(result) > _MAX_STRATEGIES:
            raise ValueError("selected_strategies exceeds maximum allowed entries")
        return result

    @model_validator(mode="after")
    def _validate_relationships(self) -> "TestsetConfigModel":
        if self.samples_per_document and self.samples_per_document > self.max_total_samples:
            raise ValueError("samples_per_document cannot exceed max_total_samples")
        return self


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(segment) for segment in error.get("loc", ())) or "<root>"
        message = error.get("msg", "invalid value")
        parts.append(f"{location}: {message}")
    return "; ".join(parts)


def validate_testset_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate *payload* and return a sanitized copy ready for hashing and persistence.

    Raises:
        ServiceError: if the supplied payload violates schema or numeric constraints.
    """

    try:
        model = TestsetConfigModel.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        message = _format_validation_error(exc)
        raise ServiceError(
            error_code=_ERROR_CODE,
            message=f"Invalid testset configuration: {message}",
            http_status=status.HTTP_400_BAD_REQUEST,
        ) from exc

    return model.model_dump(exclude_none=True)


__all__ = [
    "PersonaProfile",
    "TestsetConfigModel",
    "validate_testset_config",
]
