import collections.abc
from typing import cast, get_origin

from zodchy.codex.cqea import Context, Message

from zombus.definitions.contracts import ActorCallableContract, ActorContract
from zombus.definitions.enums import ActorKind
from zombus.definitions.errors import ActorReturnTypeError, ActorSearchTypeDerivationError

from ..internals import Actor


class ActorsRegistry:
    def __init__(self) -> None:
        self._registry: dict[type[Message] | type[Context], list[ActorContract]] = {}

    def register(self, actor_callable: ActorCallableContract) -> None:
        actor = cast(ActorContract, Actor(actor_callable))
        search_type = self._derive_search_type(actor)
        if search_type is None:
            raise ActorSearchTypeDerivationError(actor.name)
        if search_type not in self._registry:
            self._registry[search_type] = []
        self._registry[search_type].append(actor)

    def get(
        self, message_type: type[Message] | type[Context], kind: ActorKind | None = None
    ) -> collections.abc.Generator[ActorContract, None, None]:
        for actor in self._get_actors(message_type):
            if kind is None or actor.kind == kind:
                yield actor

    def _derive_search_type(self, actor: ActorContract) -> type[Message] | type[Context] | None:
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
            return message_param.type

    def __get__(self, message_type: type[Message] | type[Context]) -> list[ActorContract] | None:
        return list(self.get(message_type)) or None

    def __iter__(self) -> collections.abc.Generator[ActorContract, None, None]:
        for actors in self._registry.values():
            yield from actors

    def _get_actors(
        self, message_type: type[Message] | type[Context]
    ) -> collections.abc.Generator[ActorContract, None, None]:
        for _type in message_type.mro():
            yield from self._registry.get(_type, [])
            if _type is Message:
                break
