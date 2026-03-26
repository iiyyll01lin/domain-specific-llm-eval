"""
Evaluation run state management module.

Defines the lifecycle states for evaluation runs and valid transitions.
Implements state machine logic with validation.

Task: TASK-030a - Run Model & States
"""

from enum import Enum
from typing import Optional, Set


class RunState(str, Enum):
    """
    Evaluation run lifecycle states.
    
    State flow:
    PENDING -> RUNNING -> COMPLETED
                      -> FAILED
                      -> CANCELLED
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Define valid state transitions
VALID_TRANSITIONS: dict[RunState, Set[RunState]] = {
    RunState.PENDING: {RunState.RUNNING, RunState.CANCELLED},
    RunState.RUNNING: {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED},
    RunState.COMPLETED: set(),  # Terminal state
    RunState.FAILED: set(),  # Terminal state
    RunState.CANCELLED: set(),  # Terminal state
}


class InvalidStateTransitionError(Exception):
    """Raised when attempting an invalid state transition."""
    
    def __init__(self, from_state: RunState, to_state: RunState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid transition from {from_state.value} to {to_state.value}"
        )


def validate_transition(from_state: RunState, to_state: RunState) -> None:
    """
    Validate that a state transition is allowed.
    
    Args:
        from_state: Current state
        to_state: Target state
        
    Raises:
        InvalidStateTransitionError: If transition is not valid
        
    Example:
        >>> validate_transition(RunState.PENDING, RunState.RUNNING)
        # No exception - valid transition
        
        >>> validate_transition(RunState.COMPLETED, RunState.RUNNING)
        # Raises InvalidStateTransitionError - cannot restart completed run
    """
    valid_targets = VALID_TRANSITIONS.get(from_state, set())
    if to_state not in valid_targets:
        raise InvalidStateTransitionError(from_state, to_state)


def is_terminal_state(state: RunState) -> bool:
    """
    Check if a state is terminal (no further transitions allowed).
    
    Args:
        state: State to check
        
    Returns:
        True if state is terminal, False otherwise
    """
    return len(VALID_TRANSITIONS.get(state, set())) == 0


def get_next_valid_states(current_state: RunState) -> Set[RunState]:
    """
    Get all valid next states from the current state.
    
    Args:
        current_state: Current state
        
    Returns:
        Set of valid next states
    """
    return VALID_TRANSITIONS.get(current_state, set()).copy()
