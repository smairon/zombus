import asyncio
import collections.abc
from typing import Any

from zodchy.codex.cqea import Context, Message
from zodchy.toolbox.di import DIContainerContract, DIResolverContract
from zodchy.toolbox.processing import AsyncMessageStreamContract, AsyncProcessorContract

from zombus.definitions import enums, errors
from zombus.registration.entities import Actor
from zombus.registration.registry import ActorsRegistry

from .enums import ClusterMode
from .internals import Batch, Stream


class Processor:
    def __init__(
        self,
        actors_registry: ActorsRegistry,
        stream_filter: collections.abc.Callable[[Message], bool] | None = None,
        actors_priority: collections.abc.Sequence[enums.ActorKind] | None = None,
    ):
        """
        Initialize the processor.
        @param actors_registry: The actors registry to use.
        @param stream_filter: The stream filter for filtering message for processing by the processor.
        """
        self._actors_registry = actors_registry
        self._stream_filter = stream_filter
        self._actors_priority = actors_priority

    async def __call__(
        self, stream: AsyncMessageStreamContract, dependency_resolver: DIResolverContract
    ) -> AsyncMessageStreamContract:
        """
        Process the stream with the processor.
        Assemble the stream into batches and process the batches with the actors.
        @param stream: The stream to process.
        @param dependency_resolver: The dependency resolver to use.
        @return: The processed stream.
        """
        async for element in self._assemble_stream(stream):
            if not isinstance(element, Batch):
                yield element
                continue

            if element.message_type is None:
                continue

            async for batch in self._process_batch(element, dependency_resolver):
                for message in batch:
                    yield message

    async def _process_batch(
        self, batch: Batch, dependency_resolver: DIResolverContract
    ) -> collections.abc.AsyncGenerator[Batch, None]:
        """
        Process a batch with the actors.
        @param batch: The batch to process.
        @param dependency_resolver: The dependency resolver to use.
        @return: A generator of batches.
        """
        queue = collections.deque([batch])
        while queue:
            batch = queue.popleft()
            if batch.message_type is None:  # pragma: no cover
                yield batch
                continue
            for actor in self._get_actors_for_message_type(batch.message_type):
                async for result in self._process_batch_with_actor(
                    batch=batch, actor=actor, dependency_resolver=dependency_resolver
                ):
                    if not result:  # pragma: no cover
                        continue
                    if result.message_type is batch.message_type:
                        batch = result
                        continue
                    queue.append(result)
            yield batch

    async def _assemble_stream(
        self, stream: AsyncMessageStreamContract
    ) -> collections.abc.AsyncGenerator[Batch | Message, None]:
        """
        Assemble the stream into batches.
        @param stream: The stream to assemble into batches.
        @return: A generator of batches and messages.
        """
        current_batch = Batch()
        async for message in stream:
            if self._stream_filter and not self._stream_filter(message):
                yield message
                continue
            if current_batch.message_type is None or current_batch.message_type is type(message):
                current_batch.append(message)
                continue
            yield current_batch
            current_batch = Batch(message)
        yield current_batch

    async def _process_batch_with_actor(
        self, batch: Batch, actor: Actor, dependency_resolver: DIResolverContract
    ) -> collections.abc.AsyncGenerator[Batch, None]:
        """
        Process a batch with an actor.
        Resolve the actor and produce a new batch.
        @param batch: The batch to process.
        @param actor: The actor to process the batch with.
        @param dependency_resolver: The dependency resolver to use.
        @return: A generator of batches.
        """
        data = await self._resolve_actor(batch, actor, dependency_resolver)
        if data:
            for batch in self._collect_batches(data if isinstance(data, collections.abc.Sequence) else [data]):
                yield batch

    async def _resolve_actor(self, batch: Batch, actor: Actor, dependency_resolver: DIResolverContract) -> Any | None:
        """
        Resolve the actor.
        If the actor has a context parameter, resolve the context actor and add the context to the parameters.
        If the actor has dependency parameters, resolve the dependency parameters and add them to the parameters.
        @param batch: The batch to resolve the actor for.
        @param actor: The actor to resolve.
        @param dependency_resolver: The dependency resolver to use.
        @return: The data resolved from the actor.
        """
        if not actor.message_parameter.is_multiple and len(batch) > 1:
            raise errors.ActorParameterNotMultipleError(actor.name, actor.message_parameter.name)

        params = {}
        if actor.context_parameter:
            # Find context actor for the batch message type
            for context_actor in self._get_actors_for_context_type(actor.context_parameter.type):
                params[actor.context_parameter.name] = await self._resolve_actor(
                    batch, context_actor, dependency_resolver
                )
                # Only one context actor is allowed
                break

        if actor.dependency_parameters:
            for parameter in actor.dependency_parameters:
                params[parameter.name] = await dependency_resolver.resolve(parameter.type)

        if actor.is_async:
            result = await actor.callable(*batch, **params)  # type: ignore[misc]
        else:
            result = actor.callable(*batch, **params)
        return result

    def _get_actors_for_message_type(self, message_type: type[Message]) -> collections.abc.Generator[Actor, None, None]:
        """
        Get the actors for a message type.
        If priority is set, get the actors in the priority order, otherwise get all actors for the message type.
        @param message_type: The message type to get the actors for.
        @return: A generator of actors.
        """
        if self._actors_priority is None:
            for actor in self._actors_registry.get(message_type):
                if actor.kind != enums.ActorKind.CONTEXT:
                    yield actor
        else:
            for actor_kind in self._actors_priority:
                if actor_kind == enums.ActorKind.CONTEXT:
                    continue
                for actor in self._actors_registry.get(message_type, kind=actor_kind):
                    yield actor

    def _get_actors_for_context_type(self, context_type: type[Context]) -> collections.abc.Generator[Actor, None, None]:
        """
        Get the actors for a context type.
        @param context_type: The context type to get the actors for.
        @return: A generator of actors.
        """
        yield from self._actors_registry.get(context_type, kind=enums.ActorKind.CONTEXT)

    def _collect_batches(
        self, messages: collections.abc.Sequence[Message]
    ) -> collections.abc.Generator[Batch, None, None]:
        """
        Collect the batches from the messages.
        @param messages: The messages to collect the batches from.
        @return: A generator of batches.
        """
        if len(messages) == 1:
            if self._stream_filter is None or self._stream_filter(messages[0]):
                yield Batch(*messages)
        else:
            _map = collections.defaultdict(list)
            for message in messages:
                if self._stream_filter and not self._stream_filter(message):
                    continue
                _map[type(message)].append(message)
            for messages in _map.values():
                yield Batch(*messages)


class Cluster:
    def __init__(
        self,
        *processors: AsyncProcessorContract,
        mode: ClusterMode = ClusterMode.SEQUENTIAL,
        dependency_container: DIContainerContract | None = None,
    ):
        """
        Initialize the cluster.
        @param processors: The processors to include in the cluster.
        """
        self._processors = processors
        self._dependency_container = dependency_container
        self._mode = mode

    @property
    def has_own_dc(self) -> bool:
        """
        Get the dependency container.
        @return: The dependency container.
        """
        return self._dependency_container is not None

    async def __call__(
        self,
        stream: AsyncMessageStreamContract,
        dependency_resolver: DIResolverContract | None = None,
    ) -> AsyncMessageStreamContract:
        """
        Process the stream through the cluster.
        For each processor in the cluster, process the stream with the processor.
        @param stream: The stream to process.
        @param dependency_resolver: The dependency resolver to use.
        @return: The processed stream.
        """
        if self._dependency_container is not None:
            if self._mode == ClusterMode.SEQUENTIAL:
                async with self._dependency_container.get_resolver() as dependency_resolver:
                    common_stream = Stream([message async for message in stream])
                    for message in self._handle_results(
                        [
                            await self._handle_stream(common_stream, processor, dependency_resolver)
                            for processor in self._processors
                        ]
                    ):
                        yield message
            elif self._mode == ClusterMode.PARALLEL:
                async with self._dependency_container.get_resolver() as dependency_resolver:
                    async for message in self._parallel_handler(stream, dependency_resolver):
                        yield message

        elif dependency_resolver is not None:
            if self._mode == ClusterMode.SEQUENTIAL:
                for processor in self._processors:
                    stream = processor(stream, dependency_resolver)
                async for message in stream:
                    yield message
            elif self._mode == ClusterMode.PARALLEL:
                async for message in self._parallel_handler(stream, dependency_resolver):
                    yield message

        else:
            raise RuntimeError("No dependency resolver or container provided for cluster processing.")

    async def _parallel_handler(
        self, stream: AsyncMessageStreamContract, dependency_resolver: DIResolverContract
    ) -> AsyncMessageStreamContract:
        common_stream = Stream([message async for message in stream])
        tasks = [self._handle_stream(common_stream, processor, dependency_resolver) for processor in self._processors]
        for message in self._handle_results(await asyncio.gather(*tasks)):
            yield message

    async def _handle_stream(
        self,
        stream: Stream,
        processor: AsyncProcessorContract,
        dependency_resolver: DIResolverContract,
    ) -> list[Message]:
        """
        Run a processor with the stream and dependency resolver, collecting results before yielding.
        @param processor: The processor to run.
        @param stream: The stream to process.
        @param dependency_resolver: The dependency resolver to use.
        @return: The processed stream.
        """
        result = []
        async for message in processor(stream, dependency_resolver):  # type: ignore
            result.append(message)
        return result

    def _handle_results(self, results: list[list[Message]]) -> collections.abc.Generator[Message, None]:
        seen = set()
        for result_stream in results:
            for message in result_stream:
                if (_id := id(message)) not in seen:
                    seen.add(_id)
                    yield message
