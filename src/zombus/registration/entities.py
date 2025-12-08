import collections.abc
import inspect
from functools import cached_property
from typing import Any, get_origin

from zodchy.codex.cqea import Context, Message

from zombus.definitions import contracts, enums, errors


class Batch:
    def __init__(
        self,
        *messages: Message,
    ):
        self._messages = messages

    @cached_property
    def message_type(self) -> type[Message] | None:
        return self._derive_common_type()

    @property
    def messages(self) -> tuple[Message, ...]:
        return self._messages

    def __iter__(self) -> collections.abc.Generator[Message, None, None]:
        yield from self._messages

    def _derive_common_type(self) -> type[Message] | None:
        if len(self._messages) == 0:
            return None
        if len(self._messages) == 1:
            return type(self._messages[0])
        mros = [type(obj).mro() for obj in self._messages]
        for cls in mros[0]:
            if all(cls in mro for mro in mros[1:]):
                return cls
        # All Message subclasses share Message in their MRO, so this is unreachable
        # for valid Message objects. Included for type checker satisfaction.
        return None  # pragma: no cover


class MessageParameter:
    def __init__(
        self,
        name: str,
        type: type[Message],
        is_multiple: bool = False,
    ):
        self.name = name
        self.type = type
        self.is_multiple = is_multiple


class ContextParameter:
    def __init__(
        self,
        name: str,
        type: type[Context],
    ):
        self.name = name
        self.type = type


class DependencyParameter:
    def __init__(
        self,
        name: str,
        type: type[Any],
    ):
        self.name = name
        self.type = type


class Actor:
    def __init__(self, actor_callable: contracts.ActorCallableContract):
        self._actor_callable = actor_callable

    @cached_property
    def name(self) -> str:
        return getattr(self._actor_callable, "__name__", str(self._actor_callable))

    @cached_property
    def callable(self) -> contracts.ActorCallableContract:
        return self._actor_callable

    @cached_property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self._actor_callable)

    @cached_property
    def kind(self) -> enums.ActorKind:
        name = self.name.split("_")[-1]
        try:
            return enums.ActorKind(name)
        except ValueError:
            raise errors.UnknownActorKindError(name) from None

    @cached_property
    def message_parameter(self) -> MessageParameter:
        for param in self._parameters:
            if issubclass(param.annotation, Message):
                is_multiple = param.kind == inspect.Parameter.VAR_POSITIONAL
                return MessageParameter(param.name, param.annotation, is_multiple)
        # Unreachable: _parameters already validates that a message parameter exists
        raise errors.ActorParametersError(self._get_actor_name(), "message", "no value")  # pragma: no cover

    @cached_property
    def context_parameter(self) -> ContextParameter | None:
        for param in self._parameters:
            if issubclass(param.annotation, Context):
                return ContextParameter(param.name, param.annotation)
        return None

    @cached_property
    def dependency_parameters(self) -> list[DependencyParameter] | None:
        dependencies = []
        for param in self._parameters:
            if (
                not issubclass(param.annotation, Message)
                and not issubclass(param.annotation, Context)
                and param.annotation is not inspect.Parameter.empty
            ):
                dependencies.append(DependencyParameter(param.name, param.annotation))
        return dependencies or None

    @cached_property
    def return_type(self) -> type[Any] | None:
        return (
            self._signature.return_annotation
            if self._signature.return_annotation is not inspect.Parameter.empty
            else None
        )

    @cached_property
    def _signature(self) -> inspect.Signature:
        return inspect.signature(self._actor_callable)

    @cached_property
    def _parameters(self) -> list[inspect.Parameter]:
        parameters = []
        message_presence = False
        for param in self._signature.parameters.values():
            if param.annotation is None:
                raise errors.ActorParametersError(self._get_actor_name(), param.name, "None")
            _origin = get_origin(param.annotation)
            if _origin is not None:
                raise errors.ActorParametersError(self._get_actor_name(), param.name, _origin.__name__)
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise errors.ActorParametersError(self._get_actor_name(), param.name, "VAR_KEYWORD")
            if issubclass(param.annotation, Message):
                message_presence = True
            parameters.append(param)
        if len(parameters) == 0:
            raise errors.ActorParametersError(self._get_actor_name(), "all parameters", "0")
        if not message_presence:
            raise errors.ActorParametersError(self._get_actor_name(), "message", "no value")
        return parameters

    def _get_actor_name(self) -> str:
        return getattr(self._actor_callable, "__name__", str(self._actor_callable))
