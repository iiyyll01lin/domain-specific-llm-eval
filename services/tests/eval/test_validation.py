"""
Tests for evaluation run input validation.

Task: TASK-030b - Input Validation Layer
"""

import pytest
from pydantic import ValidationError

from services.eval.validation import (
    EvaluationRunCreateRequest,
    EvaluationRunValidator,
    InvalidProfileError,
    TestsetNotFoundError,
    get_default_profiles,
)


class TestEvaluationRunCreateRequest:
    """Test request schema validation."""
    
    def test_valid_request_minimal(self):
        """Valid minimal request passes validation."""
        request = EvaluationRunCreateRequest(
            testset_id="550e8400-e29b-41d4-a716-446655440000",
            profile="baseline",
        )
        assert request.testset_id == "550e8400-e29b-41d4-a716-446655440000"
        assert request.profile == "baseline"
        assert request.timeout_seconds == 30  # Default value
        assert request.max_retries == 3  # Default value
    
    def test_valid_request_full(self):
        """Valid request with all fields passes validation."""
        request = EvaluationRunCreateRequest(
            testset_id="550e8400-e29b-41d4-a716-446655440000",
            profile="comprehensive",
            rag_endpoint="http://localhost:8080/query",
            timeout_seconds=60,
            max_retries=5,
        )
        assert request.rag_endpoint == "http://localhost:8080/query"
        assert request.timeout_seconds == 60
        assert request.max_retries == 5
    
    def test_invalid_testset_id_too_short(self):
        """Testset ID that is too short fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRunCreateRequest(
                testset_id="550e8400",
                profile="baseline",
            )
        assert "testset_id" in str(exc_info.value)
    
    def test_invalid_testset_id_format(self):
        """Testset ID with invalid UUID format fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRunCreateRequest(
                testset_id="not-a-valid-uuid-format-here-xxxxx",
                profile="baseline",
            )
        assert "testset_id" in str(exc_info.value)
        # Either fails on length or UUID validation
        assert ("UUID" in str(exc_info.value) or "36 characters" in str(exc_info.value))
    
    def test_invalid_profile_empty(self):
        """Empty profile fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRunCreateRequest(
                testset_id="550e8400-e29b-41d4-a716-446655440000",
                profile="",
            )
        assert "profile" in str(exc_info.value)
    
    def test_invalid_profile_special_chars(self):
        """Profile with special characters fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRunCreateRequest(
                testset_id="550e8400-e29b-41d4-a716-446655440000",
                profile="baseline-profile",  # hyphen not allowed
            )
        assert "profile" in str(exc_info.value)
        assert "alphanumeric" in str(exc_info.value)
    
    def test_valid_profile_with_underscore(self):
        """Profile with underscore is valid."""
        request = EvaluationRunCreateRequest(
            testset_id="550e8400-e29b-41d4-a716-446655440000",
            profile="baseline_v2",
        )
        assert request.profile == "baseline_v2"
    
    def test_invalid_timeout_too_small(self):
        """Timeout less than 1 second fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRunCreateRequest(
                testset_id="550e8400-e29b-41d4-a716-446655440000",
                profile="baseline",
                timeout_seconds=0,
            )
        assert "timeout_seconds" in str(exc_info.value)
    
    def test_invalid_timeout_too_large(self):
        """Timeout greater than 300 seconds fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRunCreateRequest(
                testset_id="550e8400-e29b-41d4-a716-446655440000",
                profile="baseline",
                timeout_seconds=301,
            )
        assert "timeout_seconds" in str(exc_info.value)
    
    def test_invalid_max_retries_negative(self):
        """Negative max_retries fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRunCreateRequest(
                testset_id="550e8400-e29b-41d4-a716-446655440000",
                profile="baseline",
                max_retries=-1,
            )
        assert "max_retries" in str(exc_info.value)
    
    def test_invalid_max_retries_too_large(self):
        """max_retries greater than 10 fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRunCreateRequest(
                testset_id="550e8400-e29b-41d4-a716-446655440000",
                profile="baseline",
                max_retries=11,
            )
        assert "max_retries" in str(exc_info.value)


class TestEvaluationRunValidator:
    """Test evaluation run validator."""
    
    @pytest.fixture
    def existing_testset_ids(self):
        """Fixture providing set of existing testset IDs."""
        return {
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        }
    
    @pytest.fixture
    def testset_exists_fn(self, existing_testset_ids):
        """Mock function to check testset existence."""
        async def _check(testset_id: str) -> bool:
            return testset_id in existing_testset_ids
        return _check
    
    @pytest.fixture
    def validator(self, testset_exists_fn):
        """Fixture providing configured validator."""
        return EvaluationRunValidator(
            testset_exists_fn=testset_exists_fn,
            available_profiles=["baseline", "comprehensive", "fast"],
        )
    
    @pytest.mark.anyio
    async def test_valid_request_passes(self, validator):
        """Valid request with existing testset and valid profile passes."""
        request = EvaluationRunCreateRequest(
            testset_id="550e8400-e29b-41d4-a716-446655440000",
            profile="baseline",
        )
        await validator.validate(request)
        # No exception expected
    
    @pytest.mark.anyio
    async def test_missing_testset_raises_error(self, validator):
        """Request with non-existent testset raises TestsetNotFoundError."""
        request = EvaluationRunCreateRequest(
            testset_id="00000000-0000-0000-0000-000000000000",
            profile="baseline",
        )
        with pytest.raises(TestsetNotFoundError) as exc_info:
            await validator.validate(request)
        
        assert exc_info.value.testset_id == "00000000-0000-0000-0000-000000000000"
        assert exc_info.value.error_code == "testset_not_found"
        assert exc_info.value.http_status == 404
        assert "00000000-0000-0000-0000-000000000000" in exc_info.value.message
    
    @pytest.mark.anyio
    async def test_invalid_profile_raises_error(self, validator):
        """Request with invalid profile raises InvalidProfileError."""
        request = EvaluationRunCreateRequest(
            testset_id="550e8400-e29b-41d4-a716-446655440000",
            profile="nonexistent",
        )
        with pytest.raises(InvalidProfileError) as exc_info:
            await validator.validate(request)
        
        assert exc_info.value.profile == "nonexistent"
        assert exc_info.value.error_code == "invalid_profile"
        assert exc_info.value.http_status == 400
        assert "nonexistent" in exc_info.value.message
        assert "baseline" in exc_info.value.message
        assert exc_info.value.available_profiles == ["baseline", "comprehensive", "fast"]
    
    @pytest.mark.anyio
    async def test_all_valid_profiles_accepted(self, validator):
        """All configured profiles are accepted."""
        valid_profiles = ["baseline", "comprehensive", "fast"]
        
        for profile in valid_profiles:
            request = EvaluationRunCreateRequest(
                testset_id="550e8400-e29b-41d4-a716-446655440000",
                profile=profile,
            )
            await validator.validate(request)
            # No exception expected


class TestDefaultProfiles:
    """Test default profile configuration."""
    
    def test_get_default_profiles(self):
        """Default profiles list is returned."""
        profiles = get_default_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) > 0
    
    def test_default_profiles_include_baseline(self):
        """Default profiles include 'baseline'."""
        profiles = get_default_profiles()
        assert "baseline" in profiles
    
    def test_default_profiles_include_comprehensive(self):
        """Default profiles include 'comprehensive'."""
        profiles = get_default_profiles()
        assert "comprehensive" in profiles
    
    def test_default_profiles_no_duplicates(self):
        """Default profiles contain no duplicates."""
        profiles = get_default_profiles()
        assert len(profiles) == len(set(profiles))
