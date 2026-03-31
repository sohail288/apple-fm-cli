# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Generation property module for defining schema properties.

This module provides the Property class for representing properties in generation schemas,
including their types, descriptions, and associated generation guides.
"""

from typing import List, Optional, Any, Type
from .generation_guide import GenerationGuide
from .type_conversion import (
    _python_type_to_string,
)

try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise (ImportError("Python C bindings missing"))


class Property:
    """
    Represents a property in a generation schema.

    A Property defines a single field within a generation schema, including its name,
    type, optional description, and any generation guides that constrain or direct
    how the property's value should be generated.

    :ivar name: The name of the property.
    :vartype name: str
    :ivar type_class: The Python type class for this property (for example, str, int, List[str]).
    :vartype type_class: Type
    :ivar description: Optional description of the property's purpose or constraints.
    :vartype description: Optional[str]
    :ivar guides: List of generation guides that constrain how this property is generated.
    :vartype guides: List[GenerationGuide]

    Example:
        >>> from apple_fm_sdk import Property
        >>> prop = Property(
        ...     name="username",
        ...     type_class=str,
        ...     description="The user's unique identifier"
        ... )
    """

    name: str
    type_class: Type
    description: Optional[str]
    guides: List[GenerationGuide]

    def __init__(
        self,
        name: str,
        type_class: Type,
        description: Optional[str] = None,
        guides: List[GenerationGuide] = [],
    ):
        """
        Initialize a Property instance.

        :param name: The name of the property. This will be used as the key in the
            generated schema.
        :type name: str
        :param type_class: The Python type class for this property. Supported types
            include basic types (str, int, float, bool), Optional types, and List types.
        :type type_class: Type
        :param description: Optional human-readable description of the property.
            This can provide context about the property's purpose, constraints, or
            expected format.
        :type description: Optional[str]
        :param guides: List of GenerationGuide objects that constrain or direct how
            this property's value should be generated. Defaults to an empty list.
        :type guides: List[GenerationGuide]

        Example:
            >>> prop = Property(
            ...     name="age",
            ...     type_class=int,
            ...     description="Cat's age in years",
            ...     guides=[]
            ... )
        """
        self.name = name
        self.type_class = type_class
        self.description = description
        self.guides = guides

    def convert_to_c(self, schema_ptr: Any):
        """
        Convert this Property to its C representation and add it to a schema.

        This function creates a C-level property object using the Foundation Models C
        bindings, applies any associated generation guides, and adds the property to
        the specified schema pointer.

        :param schema_ptr: Pointer to the C schema object to which this property
            should be added.
        :type schema_ptr: Any
        :raises TypeError: If the property's type_class cannot be converted to a
            supported generation schema type.

        .. note::
            This is an internal function used by the schema conversion process.
            Don't call this function directly.

        .. warning::
            The schema_ptr must be a valid pointer to a C schema object created
            by the Foundation Models C bindings.
        """
        # Creates a property in C
        name_cstr = self.name.encode("utf-8")
        desc_cstr = self.description.encode("utf-8") if self.description else None

        # Verify that the type can be converted to a generation schema type
        try:
            type_name = _python_type_to_string(self.type_class)
        except TypeError as e:
            raise TypeError(
                f"Property '{self.name}' has unsupported type '{self.type_class}': {e}"
            ) from e

        # Create the property in C
        type_cstr = type_name.encode("utf-8")
        is_optional = "Optional" in str(self.type_class)
        prop_ptr = lib.FMGenerationSchemaPropertyCreate(
            name_cstr, desc_cstr, type_cstr, is_optional
        )

        # Add guides to the property
        for guide in self.guides:
            guide.convert_to_c(prop_ptr=prop_ptr)

        lib.FMGenerationSchemaAddProperty(schema_ptr, prop_ptr)
        lib.FMRelease(prop_ptr)  # Clean up property after adding
