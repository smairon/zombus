from collections.abc import Callable, Coroutine, Iterable
from typing import TypeAlias

from zodchy.codex.cqea import Message

SyncActorCallableContract: TypeAlias = Callable[..., Iterable[Message] | Message | None]
AsyncActorCallableContract: TypeAlias = Callable[..., Coroutine[None, None, Iterable[Message] | Message | None]]

ActorCallableContract: TypeAlias = SyncActorCallableContract | AsyncActorCallableContract
