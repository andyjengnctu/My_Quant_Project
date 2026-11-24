"""Read-only engineering inspection outcomes, outside scientific identities."""
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


T = TypeVar("T")


class InspectionStatus(str, Enum):
    READY = "READY"
    MISSING = "MISSING"
    INCOMPATIBLE = "INCOMPATIBLE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ArtifactInspection(Generic[T]):
    status: InspectionStatus
    reason: str
    value: T | None = None

    @property
    def ready(self) -> bool:
        return self.status is InspectionStatus.READY
