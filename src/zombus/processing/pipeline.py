from typing import Any

from zodchy.codex.cqea import Message
from zodchy.toolbox.di import DIContainerContract
from zodchy.toolbox.processing import AsyncMessageStreamContract, AsyncProcessorContract

from .internals import Stream
from .processors import Cluster


class Pipeline:
    def __init__(self, *processors: AsyncProcessorContract, dependency_container: DIContainerContract):
        """
        Initialize the pipeline.
        @param processors: The processors to include in the pipeline.
        @param dependency_container: The dependency container to use.
        """
        self._dependency_container = dependency_container
        self._processors = list(processors)

    async def __call__(self, *messages: Message, **kwargs: Any) -> AsyncMessageStreamContract:
        """
        Process the stream through the pipeline.
        For each processor in the pipeline, process the stream with the processor.
        @param stream: The stream to process.
        @param kwargs: Additional keyword arguments. (Not used, present for interface consistency)
        @return: The processed stream.
        """
        stream = Stream(list(messages))
        for processor in self._processors:
            if isinstance(processor, Cluster) and processor.has_own_dc:
                stream = Stream([m async for m in processor(stream)])
            else:
                async with self._dependency_container.get_resolver() as dependency_resolver:
                    stream = Stream([m async for m in processor(stream, dependency_resolver)])
        async for message in stream:
            yield message
