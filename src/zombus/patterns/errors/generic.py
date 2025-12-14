from dataclasses import dataclass

import zodchy


@dataclass(frozen=True, kw_only=True, slots=True)
class Error(zodchy.codex.cqea.Error):
    code: int = 500
    message: str = "An error occurred"
    details: dict[str, str] | None = None
