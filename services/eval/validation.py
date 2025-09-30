"""
Evaluation run input validation module.

Validates evaluation run creation requests including testset existence,
profile validation, and parameter constraints.

Task: TASK-030b - Input Validation Layer
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator

from services.common.errors import ServiceError


class EvaluationRunCreateRequest(BaseModel):
    """Request schema for creating an evaluation run."""
    
    testset_id: str = Field(
        ...,
        description="UUID of the testset to evaluate",
        min_length=36,
        max_length=36,
    )
    
    profile: str = Field(
        ...,
        description="Evaluation profile name (e.g., 'baseline', 'comprehensive')",
        min_length=1,
        max_length=100,
    )
    
    rag_endpoint: Optional[str] = Field(
        None,
        description="Optional override for RAG system endpoint URL",
    )
    
    timeout_seconds: Optional[int] = Field(
        30,
        description="Per-query timeout in seconds",
        ge=1,
        le=300,
    )
    
    max_retries: Optional[int] = Field(
        3,
        description="Maximum retry attempts for failed queries",
        ge=0,
        le=10,
    )
    
    @field_validator('testset_id')
    @classmethod
    def validate_testset_id_format(cls, v: str) -> str:
        """Validate testset_id is a valid UUID format."""
        # Basic UUID format check (8-4-4-4-12 hex digits)
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, v.lower()):
            raise ValueError('testset_id must be a valid UUID')
        return v
    
    @field_validator('profile')
    @classmethod
    def validate_profile_name(cls, v: str) -> str:
        """Validate profile name contains only alphanumeric and underscore."""
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('profile must contain only alphanumeric characters and underscores')
        return v


class TestsetNotFoundError(ServiceError):
    """Raised when referenced testset does not exist."""
    
    def __init__(self, testset_id: str):
        super().__init__(
            error_code="testset_not_found",
            message=f"Testset not found: {testset_id}",
            http_status=404,
        )
        self.testset_id = testset_id


class InvalidProfileError(ServiceError):
    """Raised when profile does not exist or is invalid."""
    
    def __init__(self, profile: str, available_profiles: list[str]):
        super().__init__(
            error_code="invalid_profile",
            message=f"Profile '{profile}' is not valid. Available profiles: {', '.join(available_profiles)}",
            http_status=400,
        )
        self.profile = profile
        self.available_profiles = available_profiles


class EvaluationRunValidator:
    """
    Validator for evaluation run creation requests.
    
    Validates that:
    - Referenced testset exists
    - Profile is valid
    - Parameters are within acceptable ranges
    """
    
    def __init__(
        self,
        testset_exists_fn,
        available_profiles: list[str],
    ):
        """
        Initialize validator.
        
        Args:
            testset_exists_fn: Async function that checks if testset exists
            available_profiles: List of valid profile names
        """
        self.testset_exists_fn = testset_exists_fn
        self.available_profiles = available_profiles
    
    async def validate(self, request: EvaluationRunCreateRequest) -> None:
        """
        Validate evaluation run creation request.
        
        Args:
            request: The evaluation run creation request
            
        Raises:
            TestsetNotFoundError: If testset does not exist
            InvalidProfileError: If profile is invalid
        """
        # Validate testset exists
        testset_exists = await self.testset_exists_fn(request.testset_id)
        if not testset_exists:
            raise TestsetNotFoundError(request.testset_id)
        
        # Validate profile
        if request.profile not in self.available_profiles:
            raise InvalidProfileError(request.profile, self.available_profiles)


def get_default_profiles() -> list[str]:
    """
    Get list of default evaluation profiles.
    
    Returns:
        List of available profile names
    """
    return [
        "baseline",      # Basic metrics: faithfulness, answer_relevancy
        "comprehensive", # All available metrics
        "fast",          # Minimal metrics for quick feedback
        "custom",        # User-defined metric set
    ]
