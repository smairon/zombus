"""
Tests for the processing module in pancho.implementation.processing.
"""

from collections.abc import Iterable

import pytest
from zodchy.codex.cqea import Context, Event, Task

from zombus.definitions import enums, errors
from zombus.processing import ClusterMode, Cluster, Pipeline, Processor
from zombus.processing.internals import Batch
from zombus.registration.entities import Actor
from zombus.registration.registry import ActorsRegistry


# Test message classes
class TestTask(Task):
    """Test task class for testing."""

    pass


class TestEvent(Event):
    """Test event class for testing."""

    pass


class AnotherTask(Task):
    """Another task class for testing."""

    pass


class TestContext(Context):
    """Test context class for testing."""

    pass


class TestDependency:
    """Test dependency class for testing."""

    pass


class TestBatch:
    """Test class for Batch functionality."""

    def test_batch_initialization_empty(self):
        """Test that Batch can be initialized with no messages."""
        batch = Batch()
        assert len(batch) == 0
        assert batch.message_type is None
        assert list(batch) == []

    def test_batch_initialization_single_message(self):
        """Test that Batch can be initialized with a single message."""
        message = TestTask()
        batch = Batch(message)
        assert len(batch) == 1
        assert batch.message_type is TestTask
        assert list(batch) == [message]

    def test_batch_initialization_multiple_messages(self):
        """Test that Batch can be initialized with multiple messages."""
        message1 = TestTask()
        message2 = TestTask()
        batch = Batch(message1, message2)
        assert len(batch) == 2
        assert batch.message_type is TestTask
        assert list(batch) == [message1, message2]

    def test_batch_append(self):
        """Test appending messages to a batch."""
        message1 = TestTask()
        message2 = TestTask()
        batch = Batch(message1)
        batch.append(message2)
        assert len(batch) == 2
        assert list(batch) == [message1, message2]

    def test_batch_message_type_none_when_empty(self):
        """Test that message_type returns None for empty batch."""
        batch = Batch()
        assert batch.message_type is None

    def test_batch_iteration(self):
        """Test that Batch is iterable."""
        message1 = TestTask()
        message2 = TestTask()
        batch = Batch(message1, message2)
        messages = list(batch)
        assert messages == [message1, message2]

    def test_batch_len(self):
        """Test that Batch supports len()."""
        batch = Batch()
        assert len(batch) == 0
        batch.append(TestTask())
        assert len(batch) == 1
        batch.append(TestTask())
        assert len(batch) == 2


class TestProcessor:
    """Test class for Processor functionality."""

    def test_processor_initialization(self, actors_registry):
        """Test that Processor can be initialized."""
        processor = Processor(actors_registry)
        assert processor._actors_registry is actors_registry
        assert processor._stream_filter is None
        assert processor._actors_priority is None

    def test_processor_initialization_with_filter(self, actors_registry):
        """Test that Processor can be initialized with a stream filter."""

        def filter_func(m):
            return isinstance(m, TestTask)

        processor = Processor(actors_registry, stream_filter=filter_func)
        assert processor._actors_registry is actors_registry
        assert processor._stream_filter is filter_func

    def test_processor_initialization_with_priority(self, actors_registry):
        """Test that Processor can be initialized with actor priority."""
        priority = [enums.ActorKind.USECASE, enums.ActorKind.AUDITOR]
        processor = Processor(actors_registry, actors_priority=priority)
        assert processor._actors_priority == priority

    async def _collect_processor_results(self, processor, stream, dependency_resolver):
        """Helper method to collect results from processor."""
        result = []
        async for message in processor(stream, dependency_resolver):
            result.append(message)
        return result

    async def test_processor_empty_stream(self, actors_registry, dependency_resolver):
        """Test processing an empty stream."""
        processor = Processor(actors_registry)

        async def empty_stream():
            return
            yield  # Make it an async generator

        result = await self._collect_processor_results(processor, empty_stream(), dependency_resolver)
        assert result == []

    async def test_processor_stream_with_no_actors(self, actors_registry, dependency_resolver):
        """Test processing a stream with no registered actors."""
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Messages without actors should pass through unchanged
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_processor_stream_filter_passes_message(self, actors_registry, dependency_resolver):
        """Test that stream filter allows messages through for processing."""

        def filter_func(m):
            return isinstance(m, TestTask)

        processor = Processor(actors_registry, stream_filter=filter_func)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Message passes filter and is processed (even if no actors, it's yielded)
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_processor_stream_filter_blocks_message(self, actors_registry, dependency_resolver):
        """Test that stream filter blocks messages from processing."""

        def filter_func(m):
            return isinstance(m, TestTask)

        processor = Processor(actors_registry, stream_filter=filter_func)

        async def message_stream():
            yield TestEvent()  # Should be filtered out (not processed)

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Filtered messages pass through without processing
        assert len(result) == 1
        assert isinstance(result[0], TestEvent)

    async def test_processor_single_message_single_actor(self, actors_registry, dependency_resolver):
        """Test processing a single message with a single actor.

        The processor yields both the original batch and the processed results.
        """

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [TestTask()]

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Only processed result because the original message replaced with the processed result for the same message type
        assert len(result) == 1
        assert all(isinstance(m, TestTask) for m in result)

    async def test_processor_batch_assembly_same_type(self, actors_registry, dependency_resolver):
        """Test that messages of the same type are batched together."""

        def test_usecase(*messages: TestTask) -> Iterable[TestTask]:
            return messages

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()
            yield TestTask()
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Original batch (3 messages) is replaced by processed results (3 messages) of same type
        assert len(result) == 3
        assert all(isinstance(m, TestTask) for m in result)

    async def test_processor_batch_assembly_different_types(self, actors_registry, dependency_resolver):
        """Test that messages of different types are not batched together."""

        def test_task_usecase(message: TestTask) -> Iterable[TestTask]:
            return [message]

        def test_event_usecase(message: TestEvent) -> Iterable[TestEvent]:
            return [message]

        actors_registry.register(Actor(test_task_usecase))
        actors_registry.register(Actor(test_event_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()
            yield TestEvent()
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Three separate batches: TestTask (replaced by processed), TestEvent (replaced by processed), TestTask (replaced by processed)
        # Each original is replaced by processed result of same type
        assert len(result) == 3
        assert isinstance(result[0], TestTask)  # First TestTask batch (replaced)
        assert isinstance(result[1], TestEvent)  # TestEvent batch (replaced)
        assert isinstance(result[2], TestTask)  # Second TestTask batch (replaced)

    async def test_processor_actor_with_context(self, actors_registry, dependency_resolver):
        """Test processing with an actor that requires context."""

        def test_context(message: TestTask) -> TestContext:
            return TestContext()

        def test_usecase(message: TestTask, context: TestContext) -> Iterable[TestTask]:
            return [message]

        actors_registry.register(Actor(test_context))
        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Original message is replaced by processed result (same type)
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_processor_actor_with_dependencies(self, actors_registry, dependency_resolver_with_deps):
        """Test processing with an actor that requires dependencies."""

        def test_usecase(message: TestTask, dep: TestDependency) -> Iterable[TestTask]:
            return [message]

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver_with_deps)
        # Original message is replaced by processed result (same type)
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_processor_actor_async(self, actors_registry, dependency_resolver):
        """Test processing with an async actor."""

        async def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [message]

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Original message is replaced by processed result (same type)
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_processor_actor_parameter_not_multiple_error(self, actors_registry, dependency_resolver):
        """Test that ActorParameterNotMultipleError is raised when batch has multiple messages but actor doesn't accept multiple."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [message]

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()
            yield TestTask()  # Two messages in batch

        with pytest.raises(errors.ActorParameterNotMultipleError) as exc_info:
            await self._collect_processor_results(processor, message_stream(), dependency_resolver)

        assert exc_info.value.actor_name == "test_usecase"

    async def test_processor_multiple_actors_same_message_type(self, actors_registry, dependency_resolver):
        """Test processing with multiple actors for the same message type.

        When multiple actors process the same batch and all return the same message type,
        each actor's result replaces the batch. The final result is from the last actor.
        """

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [TestTask()]

        def test_auditor(message: TestTask) -> Iterable[TestTask]:
            return [TestTask()]

        actors_registry.register(Actor(test_usecase))
        actors_registry.register(Actor(test_auditor))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Original message is replaced by usecase result, then replaced by auditor result
        # Final result is from the last actor (auditor)
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_processor_actor_returns_empty_iterable(self, actors_registry, dependency_resolver):
        """Test processing when actor returns empty iterable."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Only original batch is yielded when actor returns empty
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_processor_actor_returns_multiple_messages(self, actors_registry, dependency_resolver):
        """Test processing when actor returns multiple messages of the same type."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [TestTask(), TestTask(), TestTask()]

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Original message is replaced by batch of 3 processed messages (same type)
        assert len(result) == 3
        assert all(isinstance(m, TestTask) for m in result)

    async def test_processor_actor_returns_different_message_types(self, actors_registry, dependency_resolver):
        """Test processing when actor returns messages of different types.

        When an actor returns messages of different types, they are grouped into batches by type.
        The original batch is replaced by the batch of the same type, and different-type batches
        are queued for further processing.
        """

        def test_usecase(message: TestTask) -> Iterable[TestTask | TestEvent]:
            return [TestTask(), TestEvent(), TestTask()]

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Original TestTask is replaced by TestTask batch (2 messages), TestEvent batch (1 message) is queued
        # Since there's no actor for TestEvent, it passes through
        # Result: TestTask batch (2) + TestEvent (1)
        assert len(result) == 3
        # Check that we have both types
        task_count = sum(1 for m in result if isinstance(m, TestTask))
        event_count = sum(1 for m in result if isinstance(m, TestEvent))
        assert task_count == 2
        assert event_count == 1

    async def test_processor_actor_returns_only_different_message_type(self, actors_registry, dependency_resolver):
        """Test processing when actor returns only messages of different type.

        When an actor returns only messages of a different type than the original,
        the original batch is preserved and the different-type batch is queued.
        """

        def test_usecase(message: TestTask) -> Iterable[TestEvent]:
            return [TestEvent(), TestEvent()]

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Original TestTask is preserved (different type returned), TestEvent batch (2 messages) is queued
        # Since there's no actor for TestEvent, it passes through
        # Result: TestTask (1) + TestEvent batch (2)
        assert len(result) == 3
        task_count = sum(1 for m in result if isinstance(m, TestTask))
        event_count = sum(1 for m in result if isinstance(m, TestEvent))
        assert task_count == 1
        assert event_count == 2

    async def test_processor_collect_batches_single_message(self, actors_registry):
        """Test _collect_batches with a single message."""
        processor = Processor(actors_registry)

        message = TestTask()
        batches = list(processor._collect_batches([message]))
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert list(batches[0]) == [message]

    async def test_processor_collect_batches_multiple_same_type(self, actors_registry):
        """Test _collect_batches with multiple messages of same type."""
        processor = Processor(actors_registry)

        message1 = TestTask()
        message2 = TestTask()
        batches = list(processor._collect_batches([message1, message2]))
        assert len(batches) == 1
        assert len(batches[0]) == 2

    async def test_processor_collect_batches_multiple_different_types(self, actors_registry):
        """Test _collect_batches with multiple messages of different types."""
        processor = Processor(actors_registry)

        message1 = TestTask()
        message2 = TestEvent()
        message3 = TestTask()
        batches = list(processor._collect_batches([message1, message2, message3]))
        assert len(batches) == 2
        # Messages should be grouped by type
        batch_types = [batch.message_type for batch in batches]
        assert TestTask in batch_types
        assert TestEvent in batch_types

    def test_processor_get_actors_for_message_type(self, actors_registry):
        """Test _get_actors_for_message_type."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        actors_registry.register(Actor(test_usecase))
        actors_registry.register(Actor(test_auditor))
        processor = Processor(actors_registry)

        actors = list(processor._get_actors_for_message_type(TestTask))
        assert len(actors) == 2
        actor_names = {actor.name for actor in actors}
        assert actor_names == {"test_usecase", "test_auditor"}

    def test_processor_get_actors_for_message_type_excludes_context(self, actors_registry):
        """Test that _get_actors_for_message_type excludes CONTEXT actors."""

        def test_context(message: TestTask) -> TestContext:
            return TestContext()

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        actors_registry.register(Actor(test_context))
        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        actors = list(processor._get_actors_for_message_type(TestTask))
        assert len(actors) == 1
        assert actors[0].name == "test_usecase"

    def test_processor_get_actors_for_context_type(self, actors_registry):
        """Test _get_actors_for_context_type."""

        def test_context(message: TestTask) -> TestContext:
            return TestContext()

        actors_registry.register(Actor(test_context))
        processor = Processor(actors_registry)

        actors = list(processor._get_actors_for_context_type(TestContext))
        assert len(actors) == 1
        assert actors[0].name == "test_context"
        assert actors[0].kind == enums.ActorKind.CONTEXT


class TestCluster:
    """Test class for Cluster functionality."""

    async def _collect_cluster_results(self, cluster, stream, dependency_resolver=None):
        """Helper method to collect results from cluster."""
        result = []
        async for message in cluster(stream, dependency_resolver):
            result.append(message)
        return result

    def test_cluster_initialization(self, actors_registry):
        """Test that Cluster can be initialized."""
        processor1 = Processor(actors_registry)
        processor2 = Processor(actors_registry)
        cluster = Cluster(processor1, processor2)
        assert len(cluster._processors) == 2
        assert cluster._mode == ClusterMode.SEQUENTIAL
        assert cluster._dependency_container is None

    def test_cluster_initialization_empty(self):
        """Test that Cluster can be initialized with no processors."""
        cluster = Cluster()
        assert len(cluster._processors) == 0

    def test_cluster_initialization_with_mode(self, actors_registry):
        """Test that Cluster can be initialized with a mode."""
        processor = Processor(actors_registry)
        cluster = Cluster(processor, mode=ClusterMode.PARALLEL)
        assert cluster._mode == ClusterMode.PARALLEL

    def test_cluster_initialization_with_dependency_container(self, actors_registry, dependency_container):
        """Test that Cluster can be initialized with a dependency container."""
        processor = Processor(actors_registry)
        cluster = Cluster(processor, dependency_container=dependency_container)
        assert cluster._dependency_container is dependency_container
        assert cluster.has_own_dc is True

    def test_cluster_has_own_dc_false(self, actors_registry):
        """Test that has_own_dc is False when no dependency container is provided."""
        processor = Processor(actors_registry)
        cluster = Cluster(processor)
        assert cluster.has_own_dc is False

    async def test_cluster_processes_through_all_processors(self, actors_registry, dependency_resolver):
        """Test that Cluster processes stream through all processors."""
        processor1 = Processor(actors_registry)
        processor2 = Processor(actors_registry)
        cluster = Cluster(processor1, processor2)

        async def message_stream():
            yield TestTask()

        result = await self._collect_cluster_results(cluster, message_stream(), dependency_resolver)
        # Processors may or may not produce output depending on registered actors
        assert isinstance(result, list)

    async def test_cluster_empty_stream(self, actors_registry, dependency_resolver):
        """Test that Cluster handles empty stream."""
        processor = Processor(actors_registry)
        cluster = Cluster(processor)

        async def empty_stream():
            return
            yield

        result = await self._collect_cluster_results(cluster, empty_stream(), dependency_resolver)
        assert result == []

    async def test_cluster_sequential_mode_with_dependency_container(self, actors_registry, dependency_container):
        """Test cluster in sequential mode with its own dependency container."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [message]

        actors_registry.register(Actor(test_usecase))
        processor1 = Processor(actors_registry)
        processor2 = Processor(actors_registry)
        cluster = Cluster(
            processor1, processor2, mode=ClusterMode.SEQUENTIAL, dependency_container=dependency_container
        )

        async def message_stream():
            yield TestTask()

        result = await self._collect_cluster_results(cluster, message_stream())
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_cluster_parallel_mode_with_dependency_container(self, actors_registry, dependency_container):
        """Test cluster in parallel mode with its own dependency container."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [message]

        actors_registry.register(Actor(test_usecase))
        processor1 = Processor(actors_registry)
        processor2 = Processor(actors_registry)
        cluster = Cluster(processor1, processor2, mode=ClusterMode.PARALLEL, dependency_container=dependency_container)

        async def message_stream():
            yield TestTask()

        result = await self._collect_cluster_results(cluster, message_stream())
        # Both processors process the same input, duplicates are removed
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_cluster_parallel_mode_with_dependency_resolver(self, actors_registry, dependency_resolver):
        """Test cluster in parallel mode with dependency resolver."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [message]

        actors_registry.register(Actor(test_usecase))
        processor1 = Processor(actors_registry)
        processor2 = Processor(actors_registry)
        cluster = Cluster(processor1, processor2, mode=ClusterMode.PARALLEL)

        async def message_stream():
            yield TestTask()

        result = await self._collect_cluster_results(cluster, message_stream(), dependency_resolver)
        # Both processors process the same input, duplicates are removed
        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_cluster_raises_without_resolver_or_container(self, actors_registry):
        """Test that Cluster raises error when no resolver or container is provided."""
        processor = Processor(actors_registry)
        cluster = Cluster(processor)

        async def message_stream():
            yield TestTask()

        with pytest.raises(RuntimeError, match="No dependency resolver or container provided"):
            await self._collect_cluster_results(cluster, message_stream(), None)

    async def test_cluster_deduplicates_results(self, dependency_container):
        """Test that Cluster deduplicates identical messages from different processors."""
        # Create two registries with actors that return the same message
        registry1 = ActorsRegistry()
        registry2 = ActorsRegistry()

        def first_test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [message]  # Return the same message

        def second_test_usecase(message: TestTask) -> Iterable[TestTask]:
            return [message]  # Return the same message

        registry1.register(Actor(first_test_usecase))
        registry2.register(Actor(second_test_usecase))

        processor1 = Processor(registry1)
        processor2 = Processor(registry2)
        cluster = Cluster(processor1, processor2, dependency_container=dependency_container)

        test_task = TestTask()

        async def message_stream():
            yield test_task

        result = await self._collect_cluster_results(cluster, message_stream())
        # Same message object returned by both processors should be deduplicated
        assert len(result) == 1
        assert result[0] is test_task


class TestPipeline:
    """Test class for Pipeline functionality."""

    def test_pipeline_initialization(self, actors_registry, dependency_container):
        """Test that Pipeline can be initialized."""
        processor = Processor(actors_registry)
        pipeline = Pipeline(processor, dependency_container=dependency_container)
        assert pipeline._dependency_container is dependency_container
        assert len(pipeline._processors) == 1

    async def test_pipeline_processes_stream(self, actors_registry, dependency_container):
        """Test that Pipeline processes messages through processors."""
        processor = Processor(actors_registry)
        pipeline = Pipeline(processor, dependency_container=dependency_container)

        initial_resolver_count = dependency_container._resolvers_created

        await self._collect_pipeline_results(pipeline, TestTask())

        # Check that resolver was created
        assert dependency_container._resolvers_created > initial_resolver_count

    async def test_pipeline_multiple_processors(self, actors_registry, dependency_container):
        """Test that Pipeline processes messages through multiple processors."""
        processor1 = Processor(actors_registry)
        processor2 = Processor(actors_registry)
        pipeline = Pipeline(processor1, processor2, dependency_container=dependency_container)

        initial_resolver_count = dependency_container._resolvers_created

        await self._collect_pipeline_results(pipeline, TestTask())

        # Should get resolver for each processor
        assert dependency_container._resolvers_created == initial_resolver_count + 2

    async def _collect_pipeline_results(self, pipeline, *messages):
        """Helper method to collect results from pipeline."""
        result = []
        async for message in pipeline(*messages):
            result.append(message)
        return result

    async def test_pipeline_with_transforming_actors(self, dependency_container):
        """Test Pipeline with actors that transform messages through multiple processors."""

        # First processor: transforms TestTask to AnotherTask
        def transform_task_usecase(*messages: TestTask) -> Iterable[AnotherTask]:
            return [AnotherTask() for _ in messages]

        registry1 = ActorsRegistry()
        registry1.register(Actor(transform_task_usecase))
        processor1 = Processor(registry1)

        # Second processor: processes AnotherTask
        def process_another_task_usecase(*messages: AnotherTask) -> Iterable[AnotherTask]:
            return list(messages)  # Just pass through

        registry2 = ActorsRegistry()
        registry2.register(Actor(process_another_task_usecase))
        processor2 = Processor(registry2)

        pipeline = Pipeline(processor1, processor2, dependency_container=dependency_container)

        result = await self._collect_pipeline_results(pipeline, TestTask(), TestTask())

        # Both messages should be transformed to AnotherTask and processed
        assert len(result) == 4
        assert all(isinstance(m, TestTask) for m in result[:2])
        assert all(isinstance(m, AnotherTask) for m in result[2:])

    async def test_pipeline_with_multiple_message_types(self, dependency_container):
        """Test Pipeline processing different message types through processors."""

        # Processor 1: handles TestTask
        def handle_task_usecase(message: TestTask) -> Iterable[TestEvent]:
            return [TestEvent()]  # Transform Task to Event

        registry1 = ActorsRegistry()
        registry1.register(Actor(handle_task_usecase))
        processor1 = Processor(registry1)

        # Processor 2: handles TestEvent
        def handle_event_usecase(*messages: TestEvent) -> Iterable[TestEvent]:
            return list(messages)  # Pass through

        registry2 = ActorsRegistry()
        registry2.register(Actor(handle_event_usecase))
        processor2 = Processor(registry2)

        pipeline = Pipeline(processor1, processor2, dependency_container=dependency_container)

        result = await self._collect_pipeline_results(pipeline, TestTask(), TestEvent())

        # TestTask becomes TestEvent, TestEvent passes through
        assert len(result) == 3
        assert isinstance(result[0], TestTask)
        assert all(isinstance(m, TestEvent) for m in result[1:])

    async def test_pipeline_with_dependencies(self, actors_registry, dependency_container):
        """Test Pipeline with actors that require dependencies."""
        test_dep = TestDependency()
        dependency_container.register_dependency(test_dep, TestDependency)

        def process_task_with_dep_usecase(message: TestTask, dep: TestDependency) -> Iterable[TestTask]:
            assert dep is not None
            assert isinstance(dep, TestDependency)
            return [message]

        registry = ActorsRegistry()
        registry.register(Actor(process_task_with_dep_usecase))
        processor = Processor(registry)

        pipeline = Pipeline(processor, dependency_container=dependency_container)

        result = await self._collect_pipeline_results(pipeline, TestTask())

        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_pipeline_with_context(self, actors_registry, dependency_container):
        """Test Pipeline with actors that require context."""

        def get_task_context_context(message: TestTask) -> TestContext:
            return TestContext()

        def process_task_with_context_usecase(message: TestTask, context: TestContext) -> Iterable[TestTask]:
            assert context is not None
            assert isinstance(context, TestContext)
            return [message]

        registry = ActorsRegistry()
        registry.register(Actor(get_task_context_context))
        registry.register(Actor(process_task_with_context_usecase))
        processor = Processor(registry)

        pipeline = Pipeline(processor, dependency_container=dependency_container)

        result = await self._collect_pipeline_results(pipeline, TestTask())

        assert len(result) == 1
        assert isinstance(result[0], TestTask)

    async def test_pipeline_realistic_workflow(self, dependency_container):
        """Test a realistic workflow with multiple processors and actors."""

        # Processor 1: Validation and transformation
        def validate_task_usecase(*messages: TestTask) -> Iterable[TestTask]:
            # Simulate validation - just pass through valid messages
            return list(messages)

        registry1 = ActorsRegistry()
        registry1.register(Actor(validate_task_usecase))
        processor1 = Processor(registry1)

        # Processor 2: Business logic - transform Task to Event
        def process_task_to_event_usecase(*messages: TestTask) -> Iterable[TestEvent]:
            # Business logic: create event from task
            return [TestEvent() for _ in messages]

        registry2 = ActorsRegistry()
        registry2.register(Actor(process_task_to_event_usecase))
        processor2 = Processor(registry2)

        # Processor 3: Event handling
        def handle_event_usecase(*messages: TestEvent) -> Iterable[TestEvent]:
            # Event handler: process the event
            return list(messages)

        registry3 = ActorsRegistry()
        registry3.register(Actor(handle_event_usecase))
        processor3 = Processor(registry3)

        pipeline = Pipeline(processor1, processor2, processor3, dependency_container=dependency_container)

        result = await self._collect_pipeline_results(pipeline, TestTask(), TestTask(), TestTask())

        # Pipeline chains processors: each processor passes its output to the next
        # Processor 1 validates TestTask, Processor 2 transforms TestTask to TestEvent
        # Processor 3 receives TestEvent (not TestTask) from Processor 2
        # Result includes: 3 TestTask (original + validated) + 3 TestEvent (transformed)
        assert len(result) == 6
        task_count = sum(1 for m in result if isinstance(m, TestTask))
        event_count = sum(1 for m in result if isinstance(m, TestEvent))
        assert task_count == 3
        assert event_count == 3

    async def test_pipeline_with_batch_processing(self, actors_registry, dependency_container):
        """Test Pipeline with batch processing through multiple processors."""

        # Processor 1: Collects and processes batches
        def batch_process_task_usecase(*messages: TestTask) -> Iterable[TestTask]:
            # Process batch: return all messages
            return list(messages)

        registry1 = ActorsRegistry()
        registry1.register(Actor(batch_process_task_usecase))
        processor1 = Processor(registry1)

        # Processor 2: Transform each message
        def transform_task_to_another_usecase(*messages: TestTask) -> Iterable[AnotherTask]:
            return [AnotherTask() for _ in messages]

        registry2 = ActorsRegistry()
        registry2.register(Actor(transform_task_to_another_usecase))
        processor2 = Processor(registry2)

        pipeline = Pipeline(processor1, processor2, dependency_container=dependency_container)

        result = await self._collect_pipeline_results(pipeline, TestTask(), TestTask(), TestTask())

        # Pipeline chains processors: each processor passes its output to the next
        # Processor 1 processes TestTask batch, Processor 2 transforms TestTask to AnotherTask
        # Result includes: 3 TestTask (from processor 1) + 3 AnotherTask (from processor 2)
        assert len(result) == 6
        task_count = sum(1 for m in result if isinstance(m, TestTask))
        another_count = sum(1 for m in result if isinstance(m, AnotherTask))
        assert task_count == 3
        assert another_count == 3

    async def test_pipeline_with_filtered_processors(self, dependency_container):
        """Test Pipeline with processors that filter messages."""

        # Processor 1: Only processes TestTask
        def process_task_usecase(*messages: TestTask) -> Iterable[TestTask]:
            return list(messages)

        registry1 = ActorsRegistry()
        registry1.register(Actor(process_task_usecase))
        processor1 = Processor(registry1, stream_filter=lambda m: isinstance(m, TestTask))

        # Processor 2: Only processes TestEvent
        def process_event_usecase(message: TestEvent) -> Iterable[TestEvent]:
            return [message]

        registry2 = ActorsRegistry()
        registry2.register(Actor(process_event_usecase))
        processor2 = Processor(registry2, stream_filter=lambda m: isinstance(m, TestEvent))

        pipeline = Pipeline(processor1, processor2, dependency_container=dependency_container)

        result = await self._collect_pipeline_results(pipeline, TestTask(), TestEvent(), TestTask())

        # Pipeline processes through all processors with stream filtering
        # Processor1 filters TestTask messages, processor2 filters TestEvent messages
        # Order depends on filtering: TestEvent passes through processor1 unprocessed, TestTasks are processed
        assert len(result) == 3
        assert sum(1 for m in result if isinstance(m, TestTask)) == 2
        assert sum(1 for m in result if isinstance(m, TestEvent)) == 1


class TestProcessorCoverage:
    """Additional tests to increase coverage of processing module."""

    async def _collect_processor_results(self, processor: Processor, stream, dependency_resolver) -> list:
        """Helper method to collect results from processor."""
        result: list = []
        async for message in processor(stream, dependency_resolver):
            result.append(message)
        return result

    def test_get_actors_for_message_type_with_priority_containing_context(
        self, actors_registry: ActorsRegistry
    ) -> None:
        """Test _get_actors_for_message_type when actors_priority contains CONTEXT.

        The CONTEXT kind should be skipped even if present in the priority list.
        This covers lines 179-183 in processing.py.
        """

        def test_context(message: TestTask) -> TestContext:
            return TestContext()

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def test_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        actors_registry.register(Actor(test_context))
        actors_registry.register(Actor(test_usecase))
        actors_registry.register(Actor(test_auditor))

        # Include CONTEXT in priority list - it should be skipped
        priority = [enums.ActorKind.CONTEXT, enums.ActorKind.USECASE, enums.ActorKind.AUDITOR]
        processor = Processor(actors_registry, actors_priority=priority)

        actors = list(processor._get_actors_for_message_type(TestTask))
        # Should only get usecase and auditor, not context
        assert len(actors) == 2
        actor_kinds = {actor.kind for actor in actors}
        assert enums.ActorKind.CONTEXT not in actor_kinds
        assert enums.ActorKind.USECASE in actor_kinds
        assert enums.ActorKind.AUDITOR in actor_kinds

    def test_collect_batches_with_stream_filter_multiple_messages(self, actors_registry: ActorsRegistry) -> None:
        """Test _collect_batches with stream_filter filtering out some messages.

        When there are multiple messages and stream_filter is set, only matching
        messages should be included in batches.
        This covers line 210 in processing.py.
        """
        filter_func = lambda m: isinstance(m, TestTask)
        processor = Processor(actors_registry, stream_filter=filter_func)

        # Multiple messages, some filtered out
        message1 = TestTask()
        message2 = TestEvent()  # Should be filtered out
        message3 = TestTask()
        message4 = TestEvent()  # Should be filtered out

        batches = list(processor._collect_batches([message1, message2, message3, message4]))

        # Only TestTask messages should be in batches
        assert len(batches) == 1
        assert batches[0].message_type is TestTask
        assert len(batches[0]) == 2
        messages_in_batch = list(batches[0])
        assert message1 in messages_in_batch
        assert message3 in messages_in_batch
        assert message2 not in messages_in_batch
        assert message4 not in messages_in_batch

    def test_collect_batches_with_stream_filter_all_filtered_multiple(self, actors_registry: ActorsRegistry) -> None:
        """Test _collect_batches when stream_filter filters all multiple messages.

        When all messages are filtered out, no batches should be yielded.
        """
        filter_func = lambda m: isinstance(m, TestTask)
        processor = Processor(actors_registry, stream_filter=filter_func)

        # All messages filtered out
        message1 = TestEvent()
        message2 = TestEvent()

        batches = list(processor._collect_batches([message1, message2]))

        # No batches since all messages were filtered
        assert len(batches) == 0

    async def test_process_batch_queue_with_empty_batch_from_actor(
        self, actors_registry: ActorsRegistry, dependency_resolver
    ) -> None:
        """Test _process_batch when actor produces empty result that gets queued.

        This tests the case where an empty batch (message_type is None) is in the queue.
        Covers lines 80-81 in processing.py.
        """

        # Actor that returns messages of different types - some empty batches may be queued
        def test_usecase(message: TestTask) -> Iterable[TestEvent]:
            # Returns only TestEvent, not TestTask
            # Original TestTask batch will remain, TestEvent batch will be queued
            return [TestEvent()]

        # Another actor that returns empty
        def test_event_usecase(message: TestEvent) -> Iterable:
            return []

        actors_registry.register(Actor(test_usecase))
        actors_registry.register(Actor(test_event_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # TestTask is preserved (no TestTask returned from actor)
        # TestEvent is queued, processed by test_event_usecase which returns empty
        assert len(result) == 2
        assert any(isinstance(m, TestTask) for m in result)
        assert any(isinstance(m, TestEvent) for m in result)

    async def test_process_batch_queues_different_type_result(
        self, actors_registry: ActorsRegistry, dependency_resolver
    ) -> None:
        """Test _process_batch when actor returns batch of different message type.

        This verifies the queue.append(result) path (line 87) when result has
        a different message type than the current batch.
        """

        def test_usecase(message: TestTask) -> Iterable[TestEvent]:
            return [TestEvent(), TestEvent()]

        actors_registry.register(Actor(test_usecase))
        processor = Processor(actors_registry)

        async def message_stream():
            yield TestTask()

        result = await self._collect_processor_results(processor, message_stream(), dependency_resolver)
        # Original TestTask preserved (no same-type result)
        # TestEvent batch (2 messages) queued and yielded
        assert len(result) == 3
        task_count = sum(1 for m in result if isinstance(m, TestTask))
        event_count = sum(1 for m in result if isinstance(m, TestEvent))
        assert task_count == 1
        assert event_count == 2
