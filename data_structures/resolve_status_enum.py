from enum import StrEnum, auto


class ResolveStatusEnum(StrEnum):
    RESOLVED = auto()
    NOT_RESOLVED = auto()
    ERR_NO_GENERATION = auto()
    ERR_NO_LOGS = auto()
