from collections.abc import Callable, Generator, Iterable
from typing import Any

from zodchy.codex.cqea import View

type MappingTransformer = Callable[[Any], dict[str, Any]]


class MappingView(View):
    def __init__(self, data: Any, *transformers: MappingTransformer) -> None:
        self._data = data
        self._transformers = transformers

    def data(self) -> dict[str, Any]:
        result = self._data
        for transformer in self._transformers:
            result = transformer(result)
        if not isinstance(result, dict):
            raise TypeError("MappingView data must be a dict initially or after transformations")
        return result


class ListView(View):
    def __init__(self, stream: Iterable[Any], *transformers: MappingTransformer) -> None:
        self._stream = stream
        self._data: list[dict[str, Any]] | None = None
        self._meta: dict[str, Any] | None = None
        self._transformers = transformers

    def data(self) -> list[dict[str, Any]]:
        if self._data is None:
            if self._stream and self._transformers:
                result = self._stream
                for transformer in self._transformers:
                    result = self._apply_transformer(result, transformer)
                self._data = list(result)
            else:
                self._data = list(self._stream)
        return self._data

    def meta(self) -> dict[str, Any] | None:
        if self._meta is None:
            self._meta = {"total": len(self.data())}
        return self._meta

    def _apply_transformer(self, data: Iterable[Any], transformer: MappingTransformer) -> Generator[dict[str, Any]]:
        for item in data:
            yield transformer(item)
