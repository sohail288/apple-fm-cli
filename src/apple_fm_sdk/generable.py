# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import ctypes
import json
import logging
from typing import Any, Protocol, cast, get_args, get_origin, get_type_hints, runtime_checkable

from .c_helpers import _ManagedObject
from .generation_schema import GenerationSchema

logger = logging.getLogger(__name__)

try:
    from . import _ctypes_bindings as lib
except ImportError as e:
    raise ImportError(
        "Foundation Models C bindings not found. "
        "Please ensure _foundationmodels_ctypes.py is available."
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
            self._ptr = lib.FMGenerationIDCreate()  # type: ignore
        else:
            self._ptr = _ptr

    def __str__(self) -> str:
        """Return the string representation of the GenerationID."""
        cstr = lib.FMGenerationIDGetString(self._ptr)  # type: ignore
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
        ptr = lib.FMGeneratedContentGetID(self._ptr)  # type: ignore
        return GenerationID(_ptr=ptr)

    def value(self, type_class: type, for_property: str) -> Any:
        """Retrieve the value of a specific property from the generated content.

        :param type_class: The expected Python type of the property
        :param for_property: The name of the property to retrieve
        :return: The property value converted to the requested type
        :raises ValueError: If the property is not found or cannot be converted
        """
        data = self.to_dict()
        if for_property not in data:
            raise ValueError(f"Property '{for_property}' not found in generated content")
        return _coerce_generated_value(data[for_property], type_class)

    def to_dict(self) -> dict[str, Any]:
        """Convert the entire generated content to a dictionary."""
        jsn_ptr = lib.FMGeneratedContentGetJSONString(self._ptr)
        if not jsn_ptr:
            return {}

        json_str = str(jsn_ptr)
        result = json.loads(json_str)
        return cast(dict[str, Any], result)


def _coerce_generated_value(data: Any, type_class: type) -> Any:
    """Best-effort conversion from JSON data to the requested annotated type."""
    if type_class is Any:
        return data

    origin = get_origin(type_class)
    args = get_args(type_class)

    if origin is list and isinstance(data, list):
        item_type = args[0] if args else Any
        return [_coerce_generated_value(item, item_type) for item in data]

    if origin is not None and args:
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            return _coerce_generated_value(data, non_none_types[0])

    if type_class in (str, int, float, bool):
        return data

    if hasattr(type_class, "_generable") and getattr(type_class, "_generable") is True:
        if not isinstance(data, dict):
            raise ValueError(f"Expected object data for generable type {type_class.__name__}")

        kwargs: dict[str, Any] = {}
        type_hints = get_type_hints(type_class)
        for field_name in type_class.__dataclass_fields__:
            if field_name not in data:
                raise ValueError(f"Field '{field_name}' missing from generated content")
            field_type = type_hints.get(field_name, Any)
            kwargs[field_name] = _coerce_generated_value(data[field_name], field_type)
        return type_class(**kwargs)

    return data


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
    def PartiallyGenerated(self) -> type:  # noqa: N802
        """Get the partial generation type."""
        raise NotImplementedError("Subclasses must implement PartiallyGenerated property")
