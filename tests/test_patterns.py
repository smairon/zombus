"""
Tests for the zombus.patterns module.
"""

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

import pytest
import zodchy

from zombus.patterns.messages import (
    Command,
    Context,
    Event,
    Query,
    MappingView,
    ListView,
)
from zombus.patterns.errors import (
    Error,
    NotFoundError,
    ValidationError,
    DuplicationError,
)


# ==================== Test Classes ====================


@dataclass(frozen=True, kw_only=True, slots=True)
class CreateUserCommand(Command):
    """Test command for creating a user."""

    username: str
    email: str


@dataclass(frozen=True, kw_only=True, slots=True)
class UserCreatedEvent(Event):
    """Test event for user creation."""

    user_id: int
    username: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RequestContext(Context):
    """Test context for request."""

    request_id: str
    user_agent: str


class EmptyValue(zodchy.codex.types.NoValueType):
    """Empty value marker for query fields."""

    pass


# Singleton empty value instance
EMPTY = EmptyValue()


@dataclass(frozen=True, kw_only=True, slots=True)
class GetUsersQuery(Query):
    """Test query for getting users with filters."""

    name: zodchy.codex.operator.FilterBit[str] | EmptyValue = EMPTY
    age: zodchy.codex.operator.FilterBit[int] | EmptyValue = EMPTY
    limit: zodchy.codex.operator.SliceBit | EmptyValue = EMPTY


# ==================== Generic Message Tests ====================


class TestCommand:
    """Tests for Command class."""

    def test_command_is_frozen(self) -> None:
        """Test that Command is frozen dataclass."""
        cmd = CreateUserCommand(username="john", email="john@example.com")
        with pytest.raises(AttributeError):
            cmd.username = "jane"  # type: ignore

    def test_command_inherits_from_zodchy_command(self) -> None:
        """Test that Command inherits from zodchy.codex.cqea.Command."""
        cmd = CreateUserCommand(username="john", email="john@example.com")
        assert isinstance(cmd, zodchy.codex.cqea.Command)
        assert isinstance(cmd, zodchy.codex.cqea.Task)
        assert isinstance(cmd, zodchy.codex.cqea.Message)

    def test_command_equality(self) -> None:
        """Test that identical commands are equal."""
        cmd1 = CreateUserCommand(username="john", email="john@example.com")
        cmd2 = CreateUserCommand(username="john", email="john@example.com")
        assert cmd1 == cmd2

    def test_command_inequality(self) -> None:
        """Test that different commands are not equal."""
        cmd1 = CreateUserCommand(username="john", email="john@example.com")
        cmd2 = CreateUserCommand(username="jane", email="jane@example.com")
        assert cmd1 != cmd2


class TestEvent:
    """Tests for Event class."""

    def test_event_is_frozen(self) -> None:
        """Test that Event is frozen dataclass."""
        event = UserCreatedEvent(user_id=1, username="john")
        with pytest.raises(AttributeError):
            event.user_id = 2  # type: ignore

    def test_event_inherits_from_zodchy_event(self) -> None:
        """Test that Event inherits from zodchy.codex.cqea.Event."""
        event = UserCreatedEvent(user_id=1, username="john")
        assert isinstance(event, zodchy.codex.cqea.Event)
        assert isinstance(event, zodchy.codex.cqea.Message)

    def test_event_equality(self) -> None:
        """Test that identical events are equal."""
        event1 = UserCreatedEvent(user_id=1, username="john")
        event2 = UserCreatedEvent(user_id=1, username="john")
        assert event1 == event2


class TestContext:
    """Tests for Context class."""

    def test_context_is_frozen(self) -> None:
        """Test that Context is frozen dataclass."""
        ctx = RequestContext(request_id="req-123", user_agent="Chrome")
        with pytest.raises(AttributeError):
            ctx.request_id = "req-456"  # type: ignore

    def test_context_inherits_from_zodchy_context(self) -> None:
        """Test that Context inherits from zodchy.codex.cqea.Context."""
        ctx = RequestContext(request_id="req-123", user_agent="Chrome")
        assert isinstance(ctx, zodchy.codex.cqea.Context)


class TestQuery:
    """Tests for Query class."""

    def test_query_is_frozen(self) -> None:
        """Test that Query is frozen dataclass."""
        query = GetUsersQuery()
        with pytest.raises(AttributeError):
            query.name = "test"  # type: ignore

    def test_query_inherits_from_zodchy_query(self) -> None:
        """Test that Query inherits from zodchy.codex.cqea.Query."""
        query = GetUsersQuery()
        assert isinstance(query, zodchy.codex.cqea.Query)
        assert isinstance(query, zodchy.codex.cqea.Task)

    def test_query_iteration_empty_values(self) -> None:
        """Test that Query iteration skips Empty values."""
        query = GetUsersQuery()
        fields_list = list(query)
        assert len(fields_list) == 0

    def test_query_iteration_with_values(self) -> None:
        """Test that Query iteration yields non-Empty values."""

        class EqFilter(zodchy.codex.operator.FilterBit[str]):
            pass

        query = GetUsersQuery(name=EqFilter("john"))
        fields_list = list(query)
        assert len(fields_list) == 1
        assert fields_list[0][0] == "name"
        assert isinstance(fields_list[0][1], EqFilter)
        assert fields_list[0][1].value == "john"

    def test_query_iteration_with_multiple_values(self) -> None:
        """Test that Query iteration yields all non-Empty values."""

        class EqFilter(zodchy.codex.operator.FilterBit):
            pass

        query = GetUsersQuery(
            name=EqFilter("john"),
            age=EqFilter(30),
            limit=zodchy.codex.operator.SliceBit(10),
        )
        fields_dict = dict(query)
        assert len(fields_dict) == 3
        assert "name" in fields_dict
        assert "age" in fields_dict
        assert "limit" in fields_dict


# ==================== View Tests ====================


class TestMappingView:
    """Tests for MappingView class."""

    def test_mapping_view_creation(self) -> None:
        """Test MappingView can be created with data."""
        data = {"id": 1, "name": "John", "email": "john@example.com"}
        view = MappingView(data)
        assert view.data() == data

    def test_mapping_view_is_view(self) -> None:
        """Test MappingView inherits from View."""
        data = {"id": 1}
        view = MappingView(data)
        assert isinstance(view, zodchy.codex.cqea.View)
        assert isinstance(view, zodchy.codex.cqea.Event)

    def test_mapping_view_returns_same_dict(self) -> None:
        """Test MappingView returns the same dict instance."""
        data = {"id": 1}
        view = MappingView(data)
        assert view.data() is data

    def test_mapping_view_with_nested_data(self) -> None:
        """Test MappingView with nested dictionary."""
        data = {"user": {"id": 1, "profile": {"name": "John", "settings": {"theme": "dark"}}}}
        view = MappingView(data)
        assert view.data() == data
        assert view.data()["user"]["profile"]["settings"]["theme"] == "dark"

    def test_mapping_view_with_empty_dict(self) -> None:
        """Test MappingView with empty dictionary."""
        view = MappingView({})
        assert view.data() == {}


class TestListView:
    """Tests for ListView class."""

    def test_list_view_creation(self) -> None:
        """Test ListView can be created with iterable."""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        view = ListView(items)
        assert view.data() == items

    def test_list_view_is_view(self) -> None:
        """Test ListView inherits from View."""
        view = ListView([])
        assert isinstance(view, zodchy.codex.cqea.View)
        assert isinstance(view, zodchy.codex.cqea.Event)

    def test_list_view_with_generator(self) -> None:
        """Test ListView with generator as input."""

        def generate_items():
            for i in range(3):
                yield {"id": i}

        view = ListView(generate_items())
        result = view.data()
        assert result == [{"id": 0}, {"id": 1}, {"id": 2}]

    def test_list_view_meta(self) -> None:
        """Test ListView meta returns total count."""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        view = ListView(items)
        assert view.meta() == {"total": 3}

    def test_list_view_meta_caches_result(self) -> None:
        """Test ListView meta caches the result."""
        items = [{"id": 1}, {"id": 2}]
        view = ListView(items)
        meta1 = view.meta()
        meta2 = view.meta()
        assert meta1 is meta2

    def test_list_view_data_caches_result(self) -> None:
        """Test ListView data caches the result."""
        items = [{"id": 1}, {"id": 2}]
        view = ListView(items)
        data1 = view.data()
        data2 = view.data()
        assert data1 is data2

    def test_list_view_with_empty_list(self) -> None:
        """Test ListView with empty list."""
        view = ListView([])
        assert view.data() == []
        assert view.meta() == {"total": 0}

    def test_list_view_with_transformer(self) -> None:
        """Test ListView with single transformer."""
        items = [{"id": 1, "first_name": "John"}, {"id": 2, "first_name": "Jane"}]

        def add_full_name(item: dict[str, Any]) -> dict[str, Any]:
            return {**item, "full_name": f"{item['first_name']} Doe"}

        view = ListView(items, add_full_name)
        result = view.data()

        assert len(result) == 2
        assert result[0]["full_name"] == "John Doe"
        assert result[1]["full_name"] == "Jane Doe"

    def test_list_view_with_multiple_transformers(self) -> None:
        """Test ListView with multiple transformers applied in order."""
        items = [{"value": 1}, {"value": 2}]

        def double_value(item: dict[str, Any]) -> dict[str, Any]:
            return {**item, "value": item["value"] * 2}

        def add_label(item: dict[str, Any]) -> dict[str, Any]:
            return {**item, "label": f"Value: {item['value']}"}

        view = ListView(items, double_value, add_label)
        result = view.data()

        assert result[0]["value"] == 2
        assert result[0]["label"] == "Value: 2"
        assert result[1]["value"] == 4
        assert result[1]["label"] == "Value: 4"

    def test_list_view_transformer_with_generator(self) -> None:
        """Test ListView transformer works with generator input."""

        def generate_items():
            for i in range(3):
                yield {"id": i}

        def add_index(item: dict[str, Any]) -> dict[str, Any]:
            return {**item, "processed": True}

        view = ListView(generate_items(), add_index)
        result = view.data()

        assert len(result) == 3
        assert all(item["processed"] for item in result)

    def test_list_view_with_custom_objects(self) -> None:
        """Test ListView transformer can convert custom objects to dicts."""

        @dataclass
        class User:
            id: int
            name: str

        users = [User(id=1, name="John"), User(id=2, name="Jane")]

        def user_to_dict(user: User) -> dict[str, Any]:
            return {"id": user.id, "name": user.name, "type": "user"}

        view = ListView(users, user_to_dict)
        result = view.data()

        assert result == [
            {"id": 1, "name": "John", "type": "user"},
            {"id": 2, "name": "Jane", "type": "user"},
        ]

    def test_list_view_without_transformers_keeps_original_items(self) -> None:
        """Test ListView without transformers keeps original items."""
        items = [{"id": 1}, {"id": 2}]
        view = ListView(items)
        result = view.data()
        assert result == items

    def test_list_view_transformer_chain_order(self) -> None:
        """Test that transformers are applied in the order they are passed."""
        items = [{"step": 0}]

        def step1(item: dict[str, Any]) -> dict[str, Any]:
            return {**item, "step": 1, "history": [item["step"]]}

        def step2(item: dict[str, Any]) -> dict[str, Any]:
            return {**item, "step": 2, "history": item["history"] + [item["step"]]}

        def step3(item: dict[str, Any]) -> dict[str, Any]:
            return {**item, "step": 3, "history": item["history"] + [item["step"]]}

        view = ListView(items, step1, step2, step3)
        result = view.data()

        assert result[0]["step"] == 3
        assert result[0]["history"] == [0, 1, 2]


# ==================== Module Export Tests ====================


class TestModuleExports:
    """Tests for module exports."""

    def test_patterns_module_exports_messages(self) -> None:
        """Test that patterns module exports messages submodule."""
        from zombus import patterns

        assert hasattr(patterns, "messages")

    def test_messages_module_exports(self) -> None:
        """Test that messages module exports all expected classes."""
        from zombus.patterns import messages

        assert hasattr(messages, "Command")
        assert hasattr(messages, "Event")
        assert hasattr(messages, "Context")
        assert hasattr(messages, "Query")
        assert hasattr(messages, "MappingView")
        assert hasattr(messages, "ListView")

    def test_exported_classes_are_correct_types(self) -> None:
        """Test that exported classes are the correct types."""
        from zombus.patterns.messages import (
            Command as ExportedCommand,
            Event as ExportedEvent,
            Context as ExportedContext,
            Query as ExportedQuery,
            MappingView as ExportedMappingView,
            ListView as ExportedListView,
        )

        assert ExportedCommand is Command
        assert ExportedEvent is Event
        assert ExportedContext is Context
        assert ExportedQuery is Query
        assert ExportedMappingView is MappingView
        assert ExportedListView is ListView


# ==================== Error Tests ====================


class TestError:
    """Tests for Error class."""

    def test_error_default_values(self) -> None:
        """Test Error has correct default values."""
        error = Error()
        assert error.code == 500
        assert error.message == "An error occurred"
        assert error.details is None

    def test_error_custom_values(self) -> None:
        """Test Error can be created with custom values."""
        error = Error(code=503, message="Service unavailable", details={"reason": "maintenance"})
        assert error.code == 503
        assert error.message == "Service unavailable"
        assert error.details == {"reason": "maintenance"}

    def test_error_is_frozen(self) -> None:
        """Test that Error is frozen dataclass."""
        error = Error()
        with pytest.raises(AttributeError):
            error.code = 400  # type: ignore

    def test_error_inherits_from_zodchy_error(self) -> None:
        """Test that Error inherits from zodchy.codex.cqea.Error."""
        error = Error()
        assert isinstance(error, zodchy.codex.cqea.Error)
        assert isinstance(error, zodchy.codex.cqea.Message)

    def test_error_equality(self) -> None:
        """Test that identical errors are equal."""
        error1 = Error(code=500, message="Error")
        error2 = Error(code=500, message="Error")
        assert error1 == error2

    def test_error_inequality(self) -> None:
        """Test that different errors are not equal."""
        error1 = Error(code=500, message="Error 1")
        error2 = Error(code=500, message="Error 2")
        assert error1 != error2


class TestNotFoundError:
    """Tests for NotFoundError class."""

    def test_not_found_error_default_values(self) -> None:
        """Test NotFoundError has correct default values."""
        error = NotFoundError()
        assert error.code == 404
        assert error.message == "Resource not found"
        assert error.details is None

    def test_not_found_error_custom_message(self) -> None:
        """Test NotFoundError can have custom message."""
        error = NotFoundError(message="User not found", details={"user_id": "123"})
        assert error.code == 404
        assert error.message == "User not found"
        assert error.details == {"user_id": "123"}

    def test_not_found_error_inherits_from_error(self) -> None:
        """Test that NotFoundError inherits from Error."""
        error = NotFoundError()
        assert isinstance(error, Error)
        assert isinstance(error, zodchy.codex.cqea.Error)

    def test_not_found_error_is_frozen(self) -> None:
        """Test that NotFoundError is frozen dataclass."""
        error = NotFoundError()
        with pytest.raises(AttributeError):
            error.message = "Changed"  # type: ignore


class TestValidationError:
    """Tests for ValidationError class."""

    def test_validation_error_default_values(self) -> None:
        """Test ValidationError has correct default values."""
        error = ValidationError()
        assert error.code == 422
        assert error.message == "Validation error"
        assert error.details is None

    def test_validation_error_with_field_errors(self) -> None:
        """Test ValidationError with field validation details."""
        error = ValidationError(
            message="Invalid input", details={"email": "Invalid email format", "age": "Must be positive"}
        )
        assert error.code == 422
        assert error.message == "Invalid input"
        assert error.details == {"email": "Invalid email format", "age": "Must be positive"}

    def test_validation_error_inherits_from_error(self) -> None:
        """Test that ValidationError inherits from Error."""
        error = ValidationError()
        assert isinstance(error, Error)
        assert isinstance(error, zodchy.codex.cqea.Error)

    def test_validation_error_is_frozen(self) -> None:
        """Test that ValidationError is frozen dataclass."""
        error = ValidationError()
        with pytest.raises(AttributeError):
            error.details = {"new": "value"}  # type: ignore


class TestDuplicationError:
    """Tests for DuplicationError class."""

    def test_duplication_error_default_values(self) -> None:
        """Test DuplicationError has correct default values."""
        error = DuplicationError()
        assert error.code == 409
        assert error.message == "Resource already exists"
        assert error.details is None

    def test_duplication_error_custom_values(self) -> None:
        """Test DuplicationError with custom values."""
        error = DuplicationError(message="User already exists", details={"email": "john@example.com"})
        assert error.code == 409
        assert error.message == "User already exists"
        assert error.details == {"email": "john@example.com"}

    def test_duplication_error_inherits_from_error(self) -> None:
        """Test that DuplicationError inherits from Error."""
        error = DuplicationError()
        assert isinstance(error, Error)
        assert isinstance(error, zodchy.codex.cqea.Error)

    def test_duplication_error_is_frozen(self) -> None:
        """Test that DuplicationError is frozen dataclass."""
        error = DuplicationError()
        with pytest.raises(AttributeError):
            error.code = 500  # type: ignore


class TestErrorModuleExports:
    """Tests for errors module exports."""

    def test_errors_module_exports(self) -> None:
        """Test that errors module exports all expected classes."""
        from zombus.patterns import errors

        assert hasattr(errors, "Error")
        assert hasattr(errors, "NotFoundError")
        assert hasattr(errors, "ValidationError")
        assert hasattr(errors, "DuplicationError")

    def test_patterns_module_exports_errors(self) -> None:
        """Test that patterns module exports errors submodule."""
        from zombus import patterns

        assert hasattr(patterns, "errors")

    def test_exported_error_classes_are_correct_types(self) -> None:
        """Test that exported error classes are the correct types."""
        from zombus.patterns.errors import (
            Error as ExportedError,
            NotFoundError as ExportedNotFoundError,
            ValidationError as ExportedValidationError,
            DuplicationError as ExportedDuplicationError,
        )

        assert ExportedError is Error
        assert ExportedNotFoundError is NotFoundError
        assert ExportedValidationError is ValidationError
        assert ExportedDuplicationError is DuplicationError
