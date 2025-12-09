import collections.abc

from zodchy.codex.cqea import Message


class Batch:
    def __init__(self, *messages: Message):
        self._messages = list(messages) if messages else []

    def append(self, message: Message) -> None:
        self._messages.append(message)

    @property
    def message_type(self) -> type[Message] | None:
        return type(self._messages[0]) if self._messages else None

    def __iter__(self) -> collections.abc.Generator[Message, None, None]:
        yield from self._messages

    def __len__(self) -> int:
        return len(self._messages)


class StreamIterator(collections.abc.AsyncIterator):
    def __init__(self, messages: list[Message]):
        self._messages = messages
        self._index = 0

    async def __anext__(self) -> Message:
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        message = self._messages[self._index]
        self._index += 1
        return message


class Stream:
    def __init__(self, messages: list[Message]):
        self._messages = messages
        self._index = 0

    def extend(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    def __aiter__(self) -> StreamIterator:
        return StreamIterator(self._messages)
