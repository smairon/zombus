from enum import Enum


class ActorKind(str, Enum):
    CONTEXT = "context"
    USECASE = "usecase"
    AUDITOR = "auditor"
    WRITER = "writer"
    READER = "reader"
