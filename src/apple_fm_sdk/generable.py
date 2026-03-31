# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Guided generation support for Foundation Models SDK for Python.
This module provides classes and decorators that mirror Swift's Foundation Models
GenerationSchema, GeneratedContent, and @Generable macro functionality.
"""

from .c_helpers import _ManagedObject, _get_error_string
from .generation_schema import GenerationSchema
from .errors import GenerationErrorCode, _status_code_to_exception

import logging
from typing import (
    Any,
    Dict,
    Optional,
    Union,
    Type,
    Protocol,
    runtime_checkable,
    get_args,
)
import json
import ctypes

logger = logging.getLogger(__name__)


try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise (ImportError("Python C bindings missing"))


# MARK: GenerationID


class GenerationID:
    """Represents a unique identifier for generated content."""

    def __init__(self):
        import uuid

        self._id = str(uuid.uuid4())

    def __str__(self):
        return self._id

    def __eq__(self, other):
        return isinstance(other, GenerationID) and self._id == other._id

    def __hash__(self):
        return hash(self._id)


# MARK: Generated Content


class GeneratedContent(_ManagedObject):
    """
    Represents generated content, similar to Swift's GeneratedContent.
    This is the actual content generated according to a schema.
    """

    def __init__(
        self,
        content_dict: Optional[Dict] = None,
        id: Optional[GenerationID] = None,
        _ptr=None,
    ):
        """
        Create GeneratedContent.

        :param content_dict: Dictionary representation of the content
        :type content_dict: Optional[Dict]
        :param id: Optional GenerationID
        :type id: Optional[GenerationID]
        """
        if _ptr is not None:
            # Internal constructor for specific ptr
            super().__init__(_ptr)

            # Extract data from the C pointer if available
            if _ptr:
                # Get JSON string from C pointer
                json_cstr = lib.FMGeneratedContentGetJSONString(_ptr)
                # Check if we got a valid result
                if json_cstr and not (
                    hasattr(json_cstr, "data") and json_cstr.data is None
                ):
                    # The return value is wrapped in a String object by ctypes
                    # The String wrapper handles memory, so we don't need to manually free
                    json_str = str(json_cstr)
                    content_dict = json.loads(json_str)
                else:
                    raise ValueError("Failed to get content from C pointer")

        else:
            # Create from dictionary using C bindings
            if content_dict:
                json_str = json.dumps(content_dict).encode("utf-8")
                error_code = ctypes.c_int32()  # C error status code
                error_description = ctypes.POINTER(
                    ctypes.c_char
                )()  # C error description pointer
                ptr = lib.FMGeneratedContentCreateFromJSON(
                    json_str, ctypes.byref(error_code), ctypes.byref(error_description)
                )

                if error_code.value != GenerationErrorCode.SUCCESS:
                    # An error occurred, raise appropriate exception
                    err_code, err_desc = _get_error_string(
                        error_code, error_description
                    )
                    error_msg = "Failed to create GeneratedContent from JSON"
                    if err_desc:
                        error_msg = error_msg + ": " + err_desc
                    raise _status_code_to_exception(
                        err_code or error_code.value, error_msg
                    )

                super().__init__(ptr)

        self.id = id or GenerationID()
        self._content_dict = content_dict or {}

    @classmethod
    def from_json(cls, json_str: str) -> "GeneratedContent":
        """Create GeneratedContent from JSON string."""
        content_dict = json.loads(json_str)
        return cls(content_dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        if lib and self._ptr:
            # Use C binding to get JSON string
            json_cstr = lib.FMGeneratedContentGetJSONString(self._ptr)
            if json_cstr:
                # The return value is wrapped in a String object by ctypes
                # The String wrapper handles memory, so we don't need to manually free
                return str(json_cstr)

        # Fallback
        return json.dumps(self._content_dict)

    def value(
        self, type_class: Optional[Type] = None, for_property: Optional[str] = None
    ) -> Any:
        """
        Extract a value from the generated content.

        :param type_class: The type to convert to
        :type type_class: Optional[Type]
        :param for_property: The property name to extract (if None, extract the whole content)
        :type for_property: Optional[str]
        :return: The extracted value converted to the specified type
        :rtype: Any
        """
        if for_property:
            # Extract specific property
            raw_value = self._content_dict.get(for_property)
        else:
            # Extract whole content
            raw_value = self._content_dict

        # Handle potential nested Generable types
        if type_class:
            return self._unpack_nested_generables(type_class, raw_value, for_property)

        # Default return raw value
        return raw_value

    def _convert_value(self, value_str: str, type_class: Type) -> Any:
        """Convert a string value to the specified type."""
        if type_class is str:
            return value_str
        elif type_class is int:
            try:
                return int(value_str)
            except Exception as e:
                logger.warning(
                    f"Failed to convert '{value_str}' to int: {e}, returning 0"
                )
                return 0
        elif type_class is float:
            try:
                return float(value_str)
            except Exception as e:
                logger.warning(
                    f"Failed to convert '{value_str}' to float: {e}, returning 0.0"
                )
                return 0.0
        elif type_class is bool:
            return value_str.lower() in ("true", "1", "yes")
        elif hasattr(type_class, "__origin__") and type_class.__origin__ is list:
            # Handle List[T] types
            try:
                # First try to parse as JSON array
                return json.loads(value_str)
            except Exception as e:
                # If that fails, split by common delimiters and clean up
                logger.debug(
                    f"Failed to parse '{value_str}' as JSON list: {e}, trying delimiter split"
                )
                if "," in value_str:
                    return [
                        item.strip() for item in value_str.split(",") if item.strip()
                    ]
                elif ";" in value_str:
                    return [
                        item.strip() for item in value_str.split(";") if item.strip()
                    ]
                else:
                    # Single item - return as single-element list
                    return [value_str.strip()] if value_str.strip() else []
        else:
            # Try to parse as JSON
            try:
                return json.loads(value_str)
            except Exception as e:
                logger.debug(
                    f"Failed to parse '{value_str}' as JSON: {e}, returning as string"
                )
                return value_str

    def _unpack_nested_generables(
        self,
        type_class: Type,
        raw_value: Any,
        for_property: Optional[str] = None,
    ) -> Any:
        """Recursively unpack nested Generable types."""
        # Get outer container type if any
        origin_type = (
            type_class.__origin__
            if type_class and hasattr(type_class, "__origin__")
            else None
        )

        # Handle simple Generable type
        if isinstance(type_class, Generable):
            content = GeneratedContent(raw_value)  # Wrap raw value
            return type_class._from_generated_content(content)

        # Handle list of Generable type
        if origin_type is list:
            non_none_types = [
                arg for arg in get_args(type_class) if arg is not type(None)
            ]
            if isinstance(raw_value, list) and len(non_none_types) == 1:
                # Only one non-None type supported
                actual_type = non_none_types[0]
                # Recursively unpack inner types
                return [
                    self._unpack_nested_generables(actual_type, item, for_property)
                    for item in raw_value
                ]
            elif raw_value is None:
                return []  # Return empty list for None
            else:
                raise TypeError(
                    f"Expected list for property '{for_property}', got {type(raw_value)}"
                )

        # Handle optional type (Union[T, None])
        if origin_type is Union:
            non_none_types = [
                arg for arg in get_args(type_class) if arg is not type(None)
            ]
            # Only one non-None type supported
            if len(non_none_types) == 1:
                actual_type = non_none_types[0]
                if raw_value is None:
                    return None  # Valid since it's optional
                # Recursively unpack since it might be a Generable or list of Generables
                return self._unpack_nested_generables(
                    actual_type, raw_value, for_property
                )

        # Default return raw value, no Generable found
        return raw_value

    @property
    def is_complete(self) -> bool:
        """Check if the generated content is complete."""
        if lib and self._ptr:
            return lib.FMGeneratedContentIsComplete(self._ptr)

        # Fallback - assume complete if we have content
        return bool(self._content_dict)


# MARK: Protocols


class ConvertibleFromGeneratedContent(Protocol):
    """
    Protocol for types that can be created from GeneratedContent.
    Equivalent to Swift's ConvertibleFromGeneratedContent.
    """

    @classmethod
    def _from_generated_content(cls, content: GeneratedContent):
        """Create instance from GeneratedContent."""
        raise NotImplementedError(
            "Subclasses must implement _from_generated_content class method"
        )


class ConvertibleToGeneratedContent(Protocol):
    """
    Protocol for types that can be converted to GeneratedContent.
    Equivalent to Swift's ConvertibleToGeneratedContent.
    """

    @property
    def generated_content(self) -> GeneratedContent:
        """Convert this object to GeneratedContent."""
        raise NotImplementedError(
            "Subclasses must implement generated_content property"
        )


# MARK: Generable


@runtime_checkable
class Generable(
    ConvertibleFromGeneratedContent, ConvertibleToGeneratedContent, Protocol
):
    """
    Protocol for types that support structured generation.
    Equivalent to Swift's Generable protocol.
    """

    # Attributes for generable types
    _generable: bool = True
    _generable_description: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        raise TypeError(
            "Subclassing Protocol Generable is not allowed. "
            "Use the @fm.generable() decorator instead."
        )

    @classmethod
    def generation_schema(cls) -> GenerationSchema:
        """Get the generation schema for this type."""
        raise NotImplementedError(
            "Generable types must implement generation_schema class method"
        )

    # Default PartiallyGenerated type - can be overridden
    PartiallyGenerated = None
