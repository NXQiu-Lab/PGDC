"""Reference structure selection."""

from .selector import (
    ReferenceSelectionResult,
    ReferenceStructureSelector,
    PrototypeStructureSelector,
)

# Backward compatibility alias
PrototypeSelectionResult = ReferenceSelectionResult

__all__ = [
    "ReferenceSelectionResult",
    "ReferenceStructureSelector",
    "PrototypeSelectionResult",
    "PrototypeStructureSelector",
]
