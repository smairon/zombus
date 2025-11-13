from collections.abc import Generator, Iterable, Sequence
from enum import Enum
from typing import Any, Protocol, TypeAlias, overload

from zodchy.codex.cqea import Context, Event, Message, Task


class SyncActorCallableContract(Protocol):
    @overload
    def __call__(self, message: Message, *, context: Context, **dependencies: Any) -> Iterable[Message]:
        ...

    @overload
    def __call__(self, message: Message, **dependencies: Any) -> Iterable[Message]:
        ...

    @overload
    def __call__(self, *messages: Message, context: Context, **dependencies: Any) -> Iterable[Message]:
        ...

    @overload
    def __call__(self, *messages: Message, **dependencies: Any) -> Iterable[Message]:
        ...

    def __call__(self, *messages: Message, context: Context | None = None, **dependencies: Any) -> Iterable[Message]:
        ...


class AsyncActorCallableContract(Protocol):
    @overload
    async def __call__(self, message: Message, *, context: Context, **dependencies: Any) -> Iterable[Message]:
        ...

    @overload
    async def __call__(self, message: Message, **dependencies: Any) -> Iterable[Message]:
        ...

    @overload
    async def __call__(self, *messages: Message, context: Context, **dependencies: Any) -> Iterable[Message]:
        ...

    @overload
    async def __call__(self, *messages: Message, **dependencies: Any) -> Iterable[Message]:
        ...

    async def __call__(
        self, *messages: Message, context: Context | None = None, **dependencies: Any
    ) -> Iterable[Message]:
        ...


ActorCallableContract: TypeAlias = SyncActorCallableContract | AsyncActorCallableContract


class MessageParameter(Protocol):
    name: str
    type: type[Task | Event]
    is_multiple: bool


class ContextParameter(Protocol):
    name: str
    type: type[Context]


class DependencyParameter(Protocol):
    name: str
    type: type[Any]


class ActorContract(Protocol):
    name: str
    kind: Enum
    callable: ActorCallableContract
    is_async: bool
    message_parameter: MessageParameter
    context_parameter: ContextParameter | None
    dependency_parameters: Sequence[DependencyParameter] | None


class ActorsRegistryContract(Protocol):
    def register(self, actor_callable: ActorCallableContract) -> None:
        ...

    def get(
        self, message_type: type[Message] | type[Context], kind: Enum | None = None
    ) -> Generator[ActorContract, None, None]:
        ...

    def __get__(self, message_type: type[Message] | type[Context]) -> list[ActorContract] | None:
        ...

    def __iter__(self) -> Generator[ActorContract, None, None]:
        ...
