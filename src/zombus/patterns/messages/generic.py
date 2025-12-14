import collections.abc
from dataclasses import dataclass, fields

import zodchy


@dataclass(frozen=True, kw_only=True, slots=True)
class Command(zodchy.codex.cqea.Command):
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class Event(zodchy.codex.cqea.Event):
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class Context(zodchy.codex.cqea.Context):
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class Query(zodchy.codex.cqea.Query):
    def __iter__(self) -> collections.abc.Iterable[tuple[str, zodchy.codex.operator.ClauseBit]]:
        for field in fields(self):
            value = getattr(self, field.name)
            if zodchy.codex.types.NoValueType not in type(value).__mro__:
                yield field.name, value
