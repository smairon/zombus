from collections.abc import Callable, Coroutine, Generator, Iterable, Sequence
from enum import Enum
from typing import Any, Protocol, TypeAlias

from zodchy.codex.cqea import Context, Event, Message, Task

SyncActorCallableContract: TypeAlias = Callable[..., Iterable[Message] | Message | None]
AsyncActorCallableContract: TypeAlias = Callable[..., Coroutine[None, None, Iterable[Message] | Message | None]]

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
