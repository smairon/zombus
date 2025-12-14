from dataclasses import dataclass

from .generic import Error


@dataclass(frozen=True, kw_only=True, slots=True)
class NotFoundError(Error):
    code: int = 404
    message: str = "Resource not found"


@dataclass(frozen=True, kw_only=True, slots=True)
class ValidationError(Error):
    code: int = 422
    message: str = "Validation error"


@dataclass(frozen=True, kw_only=True, slots=True)
class DuplicationError(Error):
    code: int = 409
    message: str = "Resource already exists"
