"""Public scheduler descriptor schema facade.

Serialized schema identifiers remain versioned for backward compatibility;
Python module paths and public APIs do not.
"""
from events.semantic_descriptor import (
    AESD_SCHEMA_VERSION,
    MSSD_SCHEMA_VERSION,
    build_descriptor_object,
    json_load,
    json_save,
    normalize_slot,
)

__all__ = [
    "AESD_SCHEMA_VERSION",
    "MSSD_SCHEMA_VERSION",
    "build_descriptor_object",
    "json_load",
    "json_save",
    "normalize_slot",
]
