# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

from typing import List, Optional, Type, Any
from .generation_property import Property
from .c_helpers import _ManagedObject, _get_error_string
import ctypes
import json
from .errors import _status_code_to_exception


CPointer = ctypes._Pointer

try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise (ImportError("Python C bindings missing"))


class GenerationSchema(_ManagedObject):
    """
    Represents a generation schema that defines the structure for guided generation.

    A GenerationSchema describes the expected structure of generated content, including
    the type, properties, and nested schemas. This is used with guided generation to
    ensure the model's output conforms to a specific format, such as JSON objects with
    defined fields and types.

    The schema is similar to JSON Schema but optimized for use with Foundation Models.
    It supports basic types (str, int, float, bool), optional types, lists, and nested
    object structures.

    :ivar type_class: The Python class this schema represents.
    :vartype type_class: Type
    :ivar description: Optional human-readable description of the schema.
    :vartype description: Optional[str]
    :ivar properties: List of Property objects defining the schema's fields.
    :vartype properties: List[Property]
    :ivar nested_schemas: List of nested GenerationSchema objects for complex types.
    :vartype nested_schemas: List[GenerationSchema]
    :ivar extra_schema_fields: Additional schema fields that pass through to the
        underlying Foundation Models API.
    :vartype extra_schema_fields: dict[str, Any]

    Note: Do not instantiate GenerationSchema directly. Use the :func:`generable` decorator
    which automatically produces a GenerationSchema, or load in a json schema from your Swift app
    instead.

    .. seealso::
        :class:`Property` for defining individual schema fields.
        :class:`GenerationGuide` for constraining property values.
    """

    type_class: Type
    description: Optional[str] = None
    properties: List[Property]
    nested_schemas: List["GenerationSchema"] = []  # for nested definitions
    _ptr: CPointer

    def __init__(
        self,
        type_class: Type,
        description: Optional[str] = None,
        properties: Optional[List[Property]] = None,
        dynamic_nested_types: List["GenerationSchema"] = [],
        _ptr=None,
    ):
        """
        Initialize a GenerationSchema instance.

        :param type_class: The Python class this schema represents. This is typically
            a dataclass or similar structured type that defines the shape of the data.
        :type type_class: Type
        :param description: Optional human-readable description of what this schema
            represents. This can help provide context for the model during generation.
        :type description: Optional[str]
        :param properties: List of Property objects that define the fields/attributes
            of this schema. Each property specifies a field name, type, and optional
            constraints. If None, defaults to an empty list.
        :type properties: Optional[List[Property]]
        :param dynamic_nested_types: List of nested GenerationSchema objects for
            complex types that reference other schemas. Used when a property's type
            is itself a structured object.
        :type dynamic_nested_types: List[GenerationSchema]

        Example:
            >>> import apple_fm_sdk as fm
            >>>
            >>> @fm.generable("A cat's profile")
            >>> class Cat:
            ...     name: str
            ...     age: int
            >>>
            >>> schema = GenerationSchema(
            ...     type_class=Cat,
            ...     description="A cat's profile",
            ...     properties=[
            ...         Property(name="name", type_class=str),
            ...         Property(name="age", type_class=int)
            ...     ]
            ... )

        .. note::
            The schema is automatically converted to its C representation for use
            with the underlying Foundation Models runtime.
        """
        self.type_class = type_class
        self.description = description
        self.properties = properties or []
        self.dynamic_nested_types = dynamic_nested_types

        if _ptr is not None:
            # Internal constructor for specific C ptr
            super().__init__(_ptr)
        else:
            # Call into the C initializer
            name_cstr = type_class.__name__.encode("utf-8")
            desc_cstr = description.encode("utf-8") if description else None
            ptr = lib.FMGenerationSchemaCreate(name_cstr, desc_cstr)
            super().__init__(ptr)

            # Add referenced types to schema
            for refType in self.dynamic_nested_types:
                lib.FMGenerationSchemaAddReferenceSchema(ptr, refType._ptr)

            # Add properties to the schema
            for property in self.properties:  # Important! use self.properties here
                property.convert_to_c(schema_ptr=ptr)

    def to_dict(self) -> dict:
        """
        Convert the GenerationSchema to a dictionary representation.

        This method serializes the schema into a dictionary format compatible with
        the Foundation Models Swift API. The resulting dictionary contains the schema
        definition in a JSON-compatible format that can be used for validation,
        inspection, or transmission.

        :return: A dictionary representation of the schema, including type information,
            properties, and any nested schemas.
        :rtype: dict
        :raises ValueError: If the schema cannot be serialized (for example, empty JSON
            string returned from the C layer).
        :raises FoundationModelsError: If an error occurs during serialization in
            the underlying C/Swift implementation.

        Example:
            >>> schema = GenerationSchema(
            ...     type_class=Cat,
            ...     description="A cat's profile",
            ...     properties=[Property(name="name", type_class=str)]
            ... )
            >>> schema_dict = schema.to_dict()
            >>> print(schema_dict)
            {'name': 'Cat', 'description': 'A cat\'s profile', 'properties': [...]}

        .. note::
            This method interacts with the C layer to generate the schema dictionary.
            Memory management is handled automatically.

        .. seealso::
            The schema dictionary format follows the Foundation Models schema
            specification used by the Swift API.
        """
        error_code = ctypes.c_int32()  # C error status code
        error_description = ctypes.POINTER(
            ctypes.c_char
        )()  # C error description pointer

        jsn_string = lib.FMGenerationSchemaGetJSONString(
            self._ptr,
            ctypes.byref(error_code),
            ctypes.byref(error_description),
        )

        # Check if we got a valid result or an error
        if jsn_string is None or (
            hasattr(jsn_string, "data") and jsn_string.data is None
        ):
            # An error occurred, raise appropriate exception
            err_code, err_desc = _get_error_string(error_code, error_description)
            error_msg = "Failed to serialize GenerationSchema"
            if err_desc:
                error_msg = error_msg + ": " + err_desc
            raise _status_code_to_exception(err_code or error_code.value, error_msg)

        # It worked, parse the JSON string and return as dict
        # The return value is wrapped in a String object by ctypes
        # The String wrapper handles memory, so we don't need to manually free
        json_str = str(jsn_string)

        # Check if we got an empty string (which indicates an error)
        if not json_str or json_str.strip() == "":
            raise ValueError(
                "Failed to serialize GenerationSchema: empty JSON string returned"
            )

        result = json.loads(json_str)
        return result
