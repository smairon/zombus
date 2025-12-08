"""
Tests for the ActorsRegistry class in pancho.implementation.registration module.
"""

from collections.abc import Iterable

import pytest
from zodchy.codex.cqea import Context, Event, Task

from zombus.definitions.enums import ActorKind
from zombus.definitions.errors import ActorReturnTypeError, ActorSearchTypeDerivationError
from zombus.registration.entities import Actor
from zombus.registration.registry import ActorsRegistry


# Test message classes
class TestTask(Task):
    """Test task class for testing."""

    pass


class TestEvent(Event):
    """Test event class for testing."""

    pass


class InheritedTask(TestTask):
    """Inherited task class for testing inheritance."""

    pass


class AnotherTask(Task):
    """Another task class for testing."""

    pass


class InheritedEvent(TestEvent):
    """Inherited event class for testing inheritance."""

    pass


class TestContext(Context):
    """Test context class for testing."""

    pass


class TestActorsRegistry:
    """Test class for ActorsRegistry functionality."""

    def test_registry_initialization(self):
        """Test that ActorsRegistry can be initialized."""
        registry = ActorsRegistry()
        assert registry._registry == {}

    def test_register_single_actor(self):
        """Test registering a single actor."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))

        assert TestTask in registry._registry
        assert len(registry._registry[TestTask]) == 1
        actor = registry._registry[TestTask][0]
        assert actor.name == "test_usecase"
        assert actor.kind == ActorKind.USECASE
        assert actor.message_parameter.type is TestTask

    def test_register_multiple_actors_same_type(self):
        """Test registering multiple actors with the same message type."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_writer(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))
        registry.register(Actor(test_auditor))
        registry.register(Actor(test_writer))

        assert TestTask in registry._registry
        assert len(registry._registry[TestTask]) == 3

        actors = list(registry._registry[TestTask])
        actor_names = [actor.name for actor in actors]
        assert "test_usecase" in actor_names
        assert "test_auditor" in actor_names
        assert "test_writer" in actor_names

    def test_register_multiple_actors_different_types(self):
        """Test registering actors with different message types."""

        def test_task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_event_usecase(message: TestEvent) -> Iterable[TestEvent]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_task_usecase))
        registry.register(Actor(test_event_usecase))

        assert TestTask in registry._registry
        assert TestEvent in registry._registry
        assert len(registry._registry[TestTask]) == 1
        assert len(registry._registry[TestEvent]) == 1

    def test_get_all_actors_for_type(self):
        """Test getting all actors for a message type."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))
        registry.register(Actor(test_auditor))

        actors = list(registry.get(TestTask))
        assert len(actors) == 2
        actor_names = {actor.name for actor in actors}
        assert actor_names == {"test_usecase", "test_auditor"}

    def test_get_with_kind_filter(self):
        """Test getting actors filtered by kind."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_writer(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))
        registry.register(Actor(test_auditor))
        registry.register(Actor(test_writer))

        actors = list(registry.get(TestTask, kind=ActorKind.USECASE))
        assert len(actors) == 1
        assert actors[0].name == "test_usecase"
        assert actors[0].kind == ActorKind.USECASE

        actors = list(registry.get(TestTask, kind=ActorKind.AUDITOR))
        assert len(actors) == 1
        assert actors[0].name == "test_auditor"
        assert actors[0].kind == ActorKind.AUDITOR

    def test_get_with_kind_filter_multiple_matches(self):
        """Test getting actors with kind filter when multiple match."""

        def test1_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test2_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test1_usecase))
        registry.register(Actor(test2_usecase))

        actors = list(registry.get(TestTask, kind=ActorKind.USECASE))
        assert len(actors) == 2
        assert all(actor.kind == ActorKind.USECASE for actor in actors)

    def test_get_with_none_kind_returns_all(self):
        """Test that get with kind=None returns all actors."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))
        registry.register(Actor(test_auditor))

        actors_with_none = list(registry.get(TestTask, kind=None))
        actors_without_kind = list(registry.get(TestTask))

        assert len(actors_with_none) == 2
        assert len(actors_without_kind) == 2
        assert {actor.name for actor in actors_with_none} == {actor.name for actor in actors_without_kind}

    def test_get_with_non_existent_type(self):
        """Test getting actors for a non-existent message type."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))

        actors = list(registry.get(TestEvent))
        assert len(actors) == 0

    def test_get_with_inherited_type_mro(self):
        """Test that get uses MRO to find actors for inherited types."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))

        # Should find actors registered for parent class
        actors = list(registry.get(InheritedTask))
        assert len(actors) == 1
        assert actors[0].name == "test_usecase"

    def test_get_with_inherited_type_direct_registration(self):
        """Test getting actors when registered for inherited type."""

        def test_usecase(message: InheritedTask) -> Iterable[InheritedTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))

        # Should find actors for exact type
        actors = list(registry.get(InheritedTask))
        assert len(actors) == 1
        assert actors[0].name == "test_usecase"

    def test_get_with_inherited_type_both_parent_and_child(self):
        """Test getting actors when both parent and child types are registered."""

        def parent_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def child_usecase(message: InheritedTask) -> Iterable[InheritedTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(parent_usecase))
        registry.register(Actor(child_usecase))

        # Should find both actors
        actors = list(registry.get(InheritedTask))
        assert len(actors) == 2
        actor_names = {actor.name for actor in actors}
        assert actor_names == {"parent_usecase", "child_usecase"}

    def test_get_descriptor_method(self):
        """Test the __get__ method."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))
        registry.register(Actor(test_auditor))

        # __get__ returns a list of actors or None
        actors = registry.__get__(TestTask)
        assert actors is not None
        assert len(actors) == 2

    def test_get_descriptor_method_empty_result(self):
        """Test the __get__ descriptor method when no actors found."""

        registry = ActorsRegistry()

        actors = registry.__get__(TestTask)
        assert actors is None

    def test_iter_over_all_actors(self):
        """Test iterating over all actors in the registry."""

        def test_task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_event_usecase(message: TestEvent) -> Iterable[TestEvent]:
            return []

        def test_task_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_task_usecase))
        registry.register(Actor(test_event_usecase))
        registry.register(Actor(test_task_auditor))

        all_actors = list(registry)
        assert len(all_actors) == 3
        actor_names = {actor.name for actor in all_actors}
        assert actor_names == {"test_task_usecase", "test_event_usecase", "test_task_auditor"}

    def test_iter_empty_registry(self):
        """Test iterating over an empty registry."""

        registry = ActorsRegistry()

        all_actors = list(registry)
        assert len(all_actors) == 0

    def test_all_actor_kinds(self):
        """Test registering actors with all different kinds."""

        def test_context(message: TestTask) -> TestContext:
            return TestContext()

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_writer(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_reader(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_context))
        registry.register(Actor(test_usecase))
        registry.register(Actor(test_auditor))
        registry.register(Actor(test_writer))
        registry.register(Actor(test_reader))

        # CONTEXT actors are registered under their return type (Context), not message parameter type
        assert TestContext in registry._registry
        assert len(registry._registry[TestContext]) == 1
        assert TestTask in registry._registry
        assert len(registry._registry[TestTask]) == 4

        # Test filtering by each kind
        # CONTEXT actors are retrieved by their Context return type
        assert len(list(registry.get(TestContext, kind=ActorKind.CONTEXT))) == 1
        assert len(list(registry.get(TestTask, kind=ActorKind.USECASE))) == 1
        assert len(list(registry.get(TestTask, kind=ActorKind.AUDITOR))) == 1
        assert len(list(registry.get(TestTask, kind=ActorKind.WRITER))) == 1
        assert len(list(registry.get(TestTask, kind=ActorKind.READER))) == 1

    def test_mro_stops_at_task_base(self):
        """Test that MRO traversal stops at Task base class."""

        def test_usecase(message: Task) -> Iterable[Task]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))

        # Should find the actor for Task
        actors = list(registry.get(TestTask))
        assert len(actors) == 1
        assert actors[0].name == "test_usecase"

        # Should find the actor for InheritedTask as well
        actors = list(registry.get(InheritedTask))
        assert len(actors) == 1
        assert actors[0].name == "test_usecase"

    def test_mro_stops_at_event_base(self):
        """Test that MRO traversal stops at Event base class."""

        def test_usecase(message: Event) -> Iterable[Event]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))

        # Should find the actor for Event
        actors = list(registry.get(TestEvent))
        assert len(actors) == 1
        assert actors[0].name == "test_usecase"

        # Should find the actor for InheritedEvent as well
        actors = list(registry.get(InheritedEvent))
        assert len(actors) == 1
        assert actors[0].name == "test_usecase"

    def test_multiple_inheritance_levels(self):
        """Test MRO with multiple levels of inheritance."""

        class BaseTask(Task):
            pass

        class MiddleTask(BaseTask):
            pass

        class FinalTask(MiddleTask):
            pass

        def base_usecase(message: BaseTask) -> Iterable[BaseTask]:
            return []

        def middle_usecase(message: MiddleTask) -> Iterable[MiddleTask]:
            return []

        def final_usecase(message: FinalTask) -> Iterable[FinalTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(base_usecase))
        registry.register(Actor(middle_usecase))
        registry.register(Actor(final_usecase))

        # Should find all three actors for FinalTask
        actors = list(registry.get(FinalTask))
        assert len(actors) == 3
        actor_names = {actor.name for actor in actors}
        assert actor_names == {"base_usecase", "middle_usecase", "final_usecase"}

        # Should find base and middle for MiddleTask
        actors = list(registry.get(MiddleTask))
        assert len(actors) == 2
        actor_names = {actor.name for actor in actors}
        assert actor_names == {"base_usecase", "middle_usecase"}

        # Should find only base for BaseTask
        actors = list(registry.get(BaseTask))
        assert len(actors) == 1
        assert actors[0].name == "base_usecase"

    def test_task_and_event_separate_registries(self):
        """Test that Task and Event actors are kept separate."""

        def test_task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_event_usecase(message: TestEvent) -> Iterable[TestEvent]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_task_usecase))
        registry.register(Actor(test_event_usecase))

        # Task actors should not appear for Event queries
        task_actors = list(registry.get(TestTask))
        assert len(task_actors) == 1
        assert task_actors[0].name == "test_task_usecase"

        # Event actors should not appear for Task queries
        event_actors = list(registry.get(TestEvent))
        assert len(event_actors) == 1
        assert event_actors[0].name == "test_event_usecase"

    def test_async_actor_registration(self):
        """Test registering an async actor."""

        async def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))

        assert TestTask in registry._registry
        assert len(registry._registry[TestTask]) == 1
        actor = registry._registry[TestTask][0]
        assert actor.name == "test_usecase"
        assert actor.is_async is True

    def test_actor_with_multiple_messages_varargs(self):
        """Test registering an actor that accepts multiple messages via varargs."""

        def test_usecase(*messages: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))

        assert TestTask in registry._registry
        assert len(registry._registry[TestTask]) == 1
        actor = registry._registry[TestTask][0]
        assert actor.name == "test_usecase"
        assert actor.message_parameter.is_multiple is True

    def test_get_with_kind_filter_no_matches(self):
        """Test getting actors with kind filter when no matches."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_usecase))

        actors = list(registry.get(TestTask, kind=ActorKind.AUDITOR))
        assert len(actors) == 0

    def test_registry_with_mixed_types_and_kinds(self):
        """Test a complex scenario with multiple types and kinds."""

        def test_task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_task_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_event_usecase(message: TestEvent) -> Iterable[TestEvent]:
            return []

        def test_event_writer(message: TestEvent) -> Iterable[TestEvent]:
            return []

        def another_task_usecase(message: AnotherTask) -> Iterable[AnotherTask]:
            return []

        registry = ActorsRegistry()
        registry.register(Actor(test_task_usecase))
        registry.register(Actor(test_task_auditor))
        registry.register(Actor(test_event_usecase))
        registry.register(Actor(test_event_writer))
        registry.register(Actor(another_task_usecase))

        # Test retrieval for each type
        test_task_actors = list(registry.get(TestTask))
        assert len(test_task_actors) == 2

        test_event_actors = list(registry.get(TestEvent))
        assert len(test_event_actors) == 2

        another_task_actors = list(registry.get(AnotherTask))
        assert len(another_task_actors) == 1

        # Test filtering by kind
        test_task_usecases = list(registry.get(TestTask, kind=ActorKind.USECASE))
        assert len(test_task_usecases) == 1
        assert test_task_usecases[0].name == "test_task_usecase"

        test_event_writers = list(registry.get(TestEvent, kind=ActorKind.WRITER))
        assert len(test_event_writers) == 1
        assert test_event_writers[0].name == "test_event_writer"

        # Test iteration
        all_actors = list(registry)
        assert len(all_actors) == 5

    def test_context_actor_registration(self):
        """Test registering a CONTEXT actor with valid Context return type."""

        def test_context(message: TestTask) -> TestContext:
            return TestContext()

        registry = ActorsRegistry()
        registry.register(Actor(test_context))

        # CONTEXT actors are registered under their return type (Context), not message parameter type
        assert TestContext in registry._registry
        assert len(registry._registry[TestContext]) == 1
        assert TestTask not in registry._registry

        actor = registry._registry[TestContext][0]
        assert actor.name == "test_context"
        assert actor.kind == ActorKind.CONTEXT
        assert actor.message_parameter.type is TestTask

        # Should be retrievable by Context type
        actors = list(registry.get(TestContext, kind=ActorKind.CONTEXT))
        assert len(actors) == 1
        assert actors[0].name == "test_context"

    def test_context_actor_with_generic_return_type_raises_error(self):
        """Test that CONTEXT actor with generic return type raises ActorReturnTypeError."""

        def test_context(message: TestTask) -> Iterable[TestContext]:
            return []

        registry = ActorsRegistry()

        with pytest.raises(ActorReturnTypeError) as exc_info:
            registry.register(Actor(test_context))
        assert exc_info.value.actor_name == "test_context"

    def test_context_actor_with_non_context_return_type_raises_error(self):
        """Test that CONTEXT actor with non-Context return type raises ActorSearchTypeDerivationError."""

        def test_context(message: TestTask) -> TestTask:
            return TestTask()

        registry = ActorsRegistry()

        with pytest.raises(ActorSearchTypeDerivationError) as exc_info:
            registry.register(Actor(test_context))
        assert exc_info.value.actor_name == "test_context"

    def test_context_actor_with_inherited_context(self):
        """Test CONTEXT actor with inherited Context return type."""

        class InheritedContext(TestContext):
            pass

        def test_context(message: TestTask) -> TestContext:
            return TestContext()

        registry = ActorsRegistry()
        registry.register(Actor(test_context))

        # Should be registered under InheritedContext
        assert TestContext in registry._registry
        assert len(registry._registry[TestContext]) == 1

        # Should also be retrievable via MRO
        actors = list(registry.get(InheritedContext, kind=ActorKind.CONTEXT))
        assert len(actors) == 1
        assert actors[0].name == "test_context"
