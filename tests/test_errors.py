"""
Tests for the errors module.
"""

import pytest

from zombus.definitions.errors import (
    ActorSearchTypeDerivationError,
    ActorKindMismatchError,
    ActorReturnTypeError,
    ActorParametersError,
    ActorParameterNotMultipleError,
    UnknownActorKindError,
    ActorNotFoundError,
)


class TestActorSearchTypeDerivationError:
    """Tests for ActorSearchTypeDerivationError."""

    def test_initialization(self):
        """Test error initialization with actor name."""
        error = ActorSearchTypeDerivationError("my_actor")
        assert error.actor_name == "my_actor"
        assert "Cannot derive search type for actor my_actor" in str(error)

    def test_is_value_error(self):
        """Test that error is a ValueError."""
        error = ActorSearchTypeDerivationError("test_actor")
        assert isinstance(error, ValueError)

    def test_can_be_raised_and_caught(self):
        """Test that error can be raised and caught."""
        with pytest.raises(ActorSearchTypeDerivationError) as exc_info:
            raise ActorSearchTypeDerivationError("failing_actor")
        assert exc_info.value.actor_name == "failing_actor"


class TestActorKindMismatchError:
    """Tests for ActorKindMismatchError."""

    def test_initialization(self):
        """Test error initialization with expected and actual kinds."""
        error = ActorKindMismatchError("usecase", "context")
        assert error.expected_kind == "usecase"
        assert error.actual_kind == "context"
        assert "Actor kind mismatch: context != usecase" in str(error)

    def test_is_value_error(self):
        """Test that error is a ValueError."""
        error = ActorKindMismatchError("expected", "actual")
        assert isinstance(error, ValueError)

    def test_can_be_raised_and_caught(self):
        """Test that error can be raised and caught."""
        with pytest.raises(ActorKindMismatchError) as exc_info:
            raise ActorKindMismatchError("handler", "transformer")
        assert exc_info.value.expected_kind == "handler"
        assert exc_info.value.actual_kind == "transformer"


class TestActorReturnTypeError:
    """Tests for ActorReturnTypeError."""

    def test_initialization(self):
        """Test error initialization with actor name."""
        error = ActorReturnTypeError("bad_return_actor")
        assert error.actor_name == "bad_return_actor"
        assert "Actor bad_return_actor has a return type that is not allowed" in str(error)

    def test_is_value_error(self):
        """Test that error is a ValueError."""
        error = ActorReturnTypeError("test_actor")
        assert isinstance(error, ValueError)

    def test_can_be_raised_and_caught(self):
        """Test that error can be raised and caught."""
        with pytest.raises(ActorReturnTypeError) as exc_info:
            raise ActorReturnTypeError("invalid_actor")
        assert exc_info.value.actor_name == "invalid_actor"


class TestActorParametersError:
    """Tests for ActorParametersError."""

    def test_initialization(self):
        """Test error initialization with actor name, parameter name, and value."""
        error = ActorParametersError("my_actor", "param1", "invalid_value")
        assert error.actor_name == "my_actor"
        assert error.parameter_name == "param1"
        assert error.value == "invalid_value"
        assert "Actor my_actor has a parameter param1 with invalid_value that is not allowed" in str(error)

    def test_is_value_error(self):
        """Test that error is a ValueError."""
        error = ActorParametersError("actor", "param", "value")
        assert isinstance(error, ValueError)

    def test_can_be_raised_and_caught(self):
        """Test that error can be raised and caught."""
        with pytest.raises(ActorParametersError) as exc_info:
            raise ActorParametersError("failing_actor", "bad_param", "wrong_type")
        assert exc_info.value.actor_name == "failing_actor"
        assert exc_info.value.parameter_name == "bad_param"
        assert exc_info.value.value == "wrong_type"


class TestActorParameterNotMultipleError:
    """Tests for ActorParameterNotMultipleError."""

    def test_initialization(self):
        """Test error initialization with actor name and parameter name."""
        error = ActorParameterNotMultipleError("batch_actor", "messages")
        assert error.actor_name == "batch_actor"
        assert error.parameter_name == "messages"
        assert "Actor batch_actor has a parameter messages that is not multiple" in str(error)

    def test_is_value_error(self):
        """Test that error is a ValueError."""
        error = ActorParameterNotMultipleError("actor", "param")
        assert isinstance(error, ValueError)

    def test_can_be_raised_and_caught(self):
        """Test that error can be raised and caught."""
        with pytest.raises(ActorParameterNotMultipleError) as exc_info:
            raise ActorParameterNotMultipleError("single_actor", "items")
        assert exc_info.value.actor_name == "single_actor"
        assert exc_info.value.parameter_name == "items"


class TestUnknownActorKindError:
    """Tests for UnknownActorKindError."""

    def test_initialization(self):
        """Test error initialization with unknown kind."""
        error = UnknownActorKindError("invalid_kind")
        assert error.kind == "invalid_kind"
        assert "Unknown actor kind: invalid_kind" in str(error)

    def test_is_value_error(self):
        """Test that error is a ValueError."""
        error = UnknownActorKindError("bad_kind")
        assert isinstance(error, ValueError)

    def test_can_be_raised_and_caught(self):
        """Test that error can be raised and caught."""
        with pytest.raises(UnknownActorKindError) as exc_info:
            raise UnknownActorKindError("mystery_kind")
        assert exc_info.value.kind == "mystery_kind"


class TestActorNotFoundError:
    """Tests for ActorNotFoundError."""

    def test_initialization(self):
        """Test error initialization with message type and kind."""
        error = ActorNotFoundError("TestMessage", "usecase")
        assert error.message_type == "TestMessage"
        assert error.kind == "usecase"
        assert "Actor not found for TestMessage with kind usecase" in str(error)

    def test_is_value_error(self):
        """Test that error is a ValueError."""
        error = ActorNotFoundError("SomeMessage", "handler")
        assert isinstance(error, ValueError)

    def test_can_be_raised_and_caught(self):
        """Test that error can be raised and caught."""
        with pytest.raises(ActorNotFoundError) as exc_info:
            raise ActorNotFoundError("MissingMessage", "context")
        assert exc_info.value.message_type == "MissingMessage"
        assert exc_info.value.kind == "context"
