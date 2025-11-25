import collections.abc
from typing import get_origin

from zodchy.codex.cqea import Context, Message

from zombus.definitions.contracts import ActorCallableContract
from zombus.definitions.enums import ActorKind
from zombus.definitions.errors import ActorReturnTypeError, ActorSearchTypeDerivationError

from .internals import Actor


class ActorsRegistry:
    def __init__(self) -> None:
        self._registry: dict[type[Message], list[Actor]] = {}

    def register(self, actor_callable: ActorCallableContract) -> None:
        actor = Actor(actor_callable)
        search_type = self._derive_search_type(actor)
        if search_type is None:
            raise ActorSearchTypeDerivationError(actor.name)
        if search_type not in self._registry:
            self._registry[search_type] = []
        self._registry[search_type].append(actor)

    def get(
        self, message_type: type[Message], kind: ActorKind | None = None
    ) -> collections.abc.Generator[Actor, None, None]:
        for actor in self._get_actors(message_type):
            if kind is None or actor.kind == kind:
                yield actor

    def _derive_search_type(self, actor: Actor) -> type[Message] | None:
        if actor.kind == ActorKind.CONTEXT:
            return_type = actor.return_type
            if return_type is None:
                return None
            if get_origin(return_type):
                raise ActorReturnTypeError(actor.name)
            if hasattr(return_type, "__mro__") and Context in return_type.__mro__:
                return return_type
            return None
        else:
            message_param = actor.message_parameter
            if message_param is None:
                return None
            return message_param.type

    def __get__(self, message_type: type[Message]) -> list[Actor] | None:
        return list(self.get(message_type)) or None

    def __iter__(self) -> collections.abc.Generator[Actor, None, None]:
        for actors in self._registry.values():
            yield from actors

    def _get_actors(self, message_type: type[Message]) -> collections.abc.Generator[Actor, None, None]:
        for _type in message_type.mro():
            yield from self._registry.get(_type, [])
            if _type is Message:
                break
