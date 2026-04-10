# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import ctypes
import json
import logging
from typing import Any, Protocol, runtime_checkable

from .c_helpers import _ManagedObject
from .generation_schema import GenerationSchema

logger = logging.getLogger(__name__)

try:
    from . import _ctypes_bindings as lib
except ImportError as e:
    raise ImportError(
        "Foundation Models C bindings not found. Please ensure _foundationmodels_ctypes.py is available."
    ) from e


class GenerationID:
    """Represents a unique identifier for a generated item.

    Generation IDs are used to track items during the generation process,
    particularly when model outputs are being streamed or when dealing with
    structured outputs.
    """

    def __init__(self, _ptr: Any = None) -> None:
        """Initialize a GenerationID.

        :param _ptr: Internal C pointer to the GenerationID object
        """
        if _ptr is None:
            self._ptr = lib.FMGenerationIDCreate()
        else:
            self._ptr = _ptr

    def __str__(self) -> str:
        """Return the string representation of the GenerationID."""
        cstr = lib.FMGenerationIDGetString(self._ptr)
        return ctypes.string_at(cstr).decode("utf-8")

    def __repr__(self) -> str:
        return f"GenerationID('{self.__str__()}')"

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr:
            lib.FMRelease(self._ptr)


class GeneratedContent(_ManagedObject):
    """Represents structured content produced by a foundation model.

    GeneratedContent provides access to the model's output in a structured
    way, allowing retrieval of properties defined in a GenerationSchema.
    """

    def __init__(self, _ptr: Any) -> None:
        """Initialize with a pointer from the C library."""
        super().__init__(_ptr)

    @property
    def id(self) -> GenerationID:
        """Get the unique ID for this generated content."""
        ptr = lib.FMGeneratedContentGetID(self._ptr)
        return GenerationID(_ptr=ptr)

    def value(self, type_class: type, for_property: str) -> Any:
        """Retrieve the value of a specific property from the generated content.

        :param type_class: The expected Python type of the property
        :param for_property: The name of the property to retrieve
        :return: The property value converted to the requested type
        :raises ValueError: If the property is not found or cannot be converted
        """
        prop_bytes = for_property.encode("utf-8")
        jsn_ptr = lib.FMGeneratedContentGetPropertyJSONString(self._ptr, prop_bytes)

        if not jsn_ptr:
            raise ValueError(f"Property '{for_property}' not found in generated content")

        try:
            json_str = ctypes.string_at(jsn_ptr).decode("utf-8")
            data = json.loads(json_str)

            # Handle basic types
            if type_class in (str, int, float, bool):
                return data

            # Handle nested generables
            if hasattr(type_class, "_generable") and type_class._generable is True:
                # Create a temporary GeneratedContent for the nested object
                # This is a bit of a hack since we're re-parsing JSON
                # but the C API only gives us JSON for nested properties
                nested_ptr = lib.FMGeneratedContentCreateFromJSONString(
                    json_str.encode("utf-8")
                )
                try:
                    nested_content = GeneratedContent(_ptr=nested_ptr)
                    return type_class._from_generated_content(nested_content)
                finally:
                    lib.FMRelease(nested_ptr)

            return data
        finally:
            lib.FMRelease(jsn_ptr)

    def to_dict(self) -> dict[str, Any]:
        """Convert the entire generated content to a dictionary."""
        jsn_ptr = lib.FMGeneratedContentGetJSONString(self._ptr)
        if not jsn_ptr:
            return {}

        try:
            json_str = ctypes.string_at(jsn_ptr).decode("utf-8")
            return json.loads(json_str)
        finally:
            lib.FMRelease(jsn_ptr)


# MARK: Protocols


@runtime_checkable
class ConvertibleFromGeneratedContent(Protocol):
    """Protocol for types that can be created from GeneratedContent.

    Equivalent to Swift's ConvertibleFromGeneratedContent.
    """

    @classmethod
    def _from_generated_content(cls, content: GeneratedContent) -> Any:
        """Create instance from GeneratedContent."""
        raise NotImplementedError("Subclasses must implement _from_generated_content class method")


@runtime_checkable
class ConvertibleToGeneratedContent(Protocol):
    """Protocol for types that can be converted to GeneratedContent.

    Equivalent to Swift's ConvertibleToGeneratedContent.
    """

    def generated_content(self) -> GeneratedContent:
        """Convert this instance to GeneratedContent."""
        raise NotImplementedError("Subclasses must implement generated_content method")


# MARK: Generable


@runtime_checkable
class Generable(ConvertibleFromGeneratedContent, ConvertibleToGeneratedContent, Protocol):
    """Type representing a class that can be used for guided generation.

    A Generable type must implement both ``ConvertibleFromGeneratedContent``
    and ``ConvertibleToGeneratedContent`` protocols. This is typically
    achieved by applying the ``@generable`` decorator to a dataclass.
    """

    _generable: bool
    _generable_description: str | None

    @classmethod
    def generation_schema(cls) -> GenerationSchema:
        """Generate a schema for this type."""
        raise NotImplementedError("Subclasses must implement generation_schema class method")

    @property
    def PartiallyGenerated(self) -> type:
        """Get the partial generation type."""
        raise NotImplementedError("Subclasses must implement PartiallyGenerated property")
