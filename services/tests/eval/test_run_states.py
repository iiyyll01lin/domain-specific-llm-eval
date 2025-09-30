"""
Tests for evaluation run state management.

Task: TASK-030a - Run Model & States
"""

import pytest

from services.eval.run_states import (
    InvalidStateTransitionError,
    RunState,
    VALID_TRANSITIONS,
    get_next_valid_states,
    is_terminal_state,
    validate_transition,
)


class TestRunState:
    """Test RunState enum."""
    
    def test_all_states_defined(self):
        """All required states are defined."""
        assert RunState.PENDING == "pending"
        assert RunState.RUNNING == "running"
        assert RunState.COMPLETED == "completed"
        assert RunState.FAILED == "failed"
        assert RunState.CANCELLED == "cancelled"
    
    def test_state_count(self):
        """Verify expected number of states."""
        assert len(RunState) == 5


class TestStateTransitions:
    """Test state transition validation."""
    
    def test_valid_transition_pending_to_running(self):
        """PENDING -> RUNNING is valid."""
        validate_transition(RunState.PENDING, RunState.RUNNING)
        # No exception expected
    
    def test_valid_transition_pending_to_cancelled(self):
        """PENDING -> CANCELLED is valid."""
        validate_transition(RunState.PENDING, RunState.CANCELLED)
        # No exception expected
    
    def test_valid_transition_running_to_completed(self):
        """RUNNING -> COMPLETED is valid."""
        validate_transition(RunState.RUNNING, RunState.COMPLETED)
        # No exception expected
    
    def test_valid_transition_running_to_failed(self):
        """RUNNING -> FAILED is valid."""
        validate_transition(RunState.RUNNING, RunState.FAILED)
        # No exception expected
    
    def test_valid_transition_running_to_cancelled(self):
        """RUNNING -> CANCELLED is valid."""
        validate_transition(RunState.RUNNING, RunState.CANCELLED)
        # No exception expected
    
    def test_invalid_transition_completed_to_running(self):
        """COMPLETED -> RUNNING is invalid."""
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            validate_transition(RunState.COMPLETED, RunState.RUNNING)
        
        assert exc_info.value.from_state == RunState.COMPLETED
        assert exc_info.value.to_state == RunState.RUNNING
        assert "completed" in str(exc_info.value)
        assert "running" in str(exc_info.value)
    
    def test_invalid_transition_failed_to_running(self):
        """FAILED -> RUNNING is invalid."""
        with pytest.raises(InvalidStateTransitionError):
            validate_transition(RunState.FAILED, RunState.RUNNING)
    
    def test_invalid_transition_cancelled_to_running(self):
        """CANCELLED -> RUNNING is invalid."""
        with pytest.raises(InvalidStateTransitionError):
            validate_transition(RunState.CANCELLED, RunState.RUNNING)
    
    def test_invalid_transition_pending_to_completed(self):
        """PENDING -> COMPLETED is invalid (must go through RUNNING)."""
        with pytest.raises(InvalidStateTransitionError):
            validate_transition(RunState.PENDING, RunState.COMPLETED)
    
    def test_invalid_transition_pending_to_failed(self):
        """PENDING -> FAILED is invalid (must go through RUNNING)."""
        with pytest.raises(InvalidStateTransitionError):
            validate_transition(RunState.PENDING, RunState.FAILED)


class TestTerminalStates:
    """Test terminal state detection."""
    
    def test_completed_is_terminal(self):
        """COMPLETED is a terminal state."""
        assert is_terminal_state(RunState.COMPLETED) is True
    
    def test_failed_is_terminal(self):
        """FAILED is a terminal state."""
        assert is_terminal_state(RunState.FAILED) is True
    
    def test_cancelled_is_terminal(self):
        """CANCELLED is a terminal state."""
        assert is_terminal_state(RunState.CANCELLED) is True
    
    def test_pending_is_not_terminal(self):
        """PENDING is not a terminal state."""
        assert is_terminal_state(RunState.PENDING) is False
    
    def test_running_is_not_terminal(self):
        """RUNNING is not a terminal state."""
        assert is_terminal_state(RunState.RUNNING) is False


class TestGetNextValidStates:
    """Test getting next valid states."""
    
    def test_next_states_from_pending(self):
        """Get next valid states from PENDING."""
        next_states = get_next_valid_states(RunState.PENDING)
        assert next_states == {RunState.RUNNING, RunState.CANCELLED}
    
    def test_next_states_from_running(self):
        """Get next valid states from RUNNING."""
        next_states = get_next_valid_states(RunState.RUNNING)
        assert next_states == {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}
    
    def test_next_states_from_completed(self):
        """Get next valid states from COMPLETED (none)."""
        next_states = get_next_valid_states(RunState.COMPLETED)
        assert next_states == set()
    
    def test_next_states_from_failed(self):
        """Get next valid states from FAILED (none)."""
        next_states = get_next_valid_states(RunState.FAILED)
        assert next_states == set()
    
    def test_next_states_from_cancelled(self):
        """Get next valid states from CANCELLED (none)."""
        next_states = get_next_valid_states(RunState.CANCELLED)
        assert next_states == set()
    
    def test_returned_set_is_copy(self):
        """Returned set is a copy (mutations don't affect original)."""
        next_states = get_next_valid_states(RunState.PENDING)
        next_states.add(RunState.FAILED)  # Modify returned set
        
        # Original VALID_TRANSITIONS should be unchanged
        assert RunState.FAILED not in VALID_TRANSITIONS[RunState.PENDING]


class TestTransitionMatrix:
    """Test complete transition matrix for consistency."""
    
    def test_all_states_have_transitions_defined(self):
        """All states have entries in VALID_TRANSITIONS."""
        for state in RunState:
            assert state in VALID_TRANSITIONS, f"State {state} missing from VALID_TRANSITIONS"
    
    def test_terminal_states_have_no_transitions(self):
        """Terminal states have empty transition sets."""
        terminal_states = {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}
        for state in terminal_states:
            assert len(VALID_TRANSITIONS[state]) == 0, f"Terminal state {state} has transitions"
    
    def test_non_terminal_states_have_transitions(self):
        """Non-terminal states have at least one valid transition."""
        non_terminal_states = {RunState.PENDING, RunState.RUNNING}
        for state in non_terminal_states:
            assert len(VALID_TRANSITIONS[state]) > 0, f"Non-terminal state {state} has no transitions"
