class ActorSearchTypeDerivationError(ValueError):
    def __init__(self, actor_name: str):
        self.actor_name = actor_name
        super().__init__(f"Cannot derive search type for actor {actor_name}")


class ActorKindMismatchError(ValueError):
    def __init__(self, expected_kind: str, actual_kind: str):
        self.expected_kind = expected_kind
        self.actual_kind = actual_kind
        super().__init__(f"Actor kind mismatch: {actual_kind} != {expected_kind}")


class ActorReturnTypeError(ValueError):
    def __init__(self, actor_name: str):
        self.actor_name = actor_name
        super().__init__(f"Actor {actor_name} has a return type that is not allowed")


class ActorParametersError(ValueError):
    def __init__(self, actor_name: str, parameter_name: str, value: str):
        self.actor_name = actor_name
        self.parameter_name = parameter_name
        self.value = value
        super().__init__(f"Actor {actor_name} has a parameter {parameter_name} with {value} that is not allowed")


class ActorParameterNotMultipleError(ValueError):
    def __init__(self, actor_name: str, parameter_name: str):
        self.actor_name = actor_name
        self.parameter_name = parameter_name
        super().__init__(f"Actor {actor_name} has a parameter {parameter_name} that is not multiple")


class UnknownActorKindError(ValueError):
    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(f"Unknown actor kind: {kind}")


class ActorNotFoundError(ValueError):
    def __init__(self, message_type: str, kind: str):
        self.message_type = message_type
        self.kind = kind
        super().__init__(f"Actor not found for {message_type} with kind {kind}")
