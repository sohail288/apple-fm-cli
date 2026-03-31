# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
This module provides utilities for converting Python types to their corresponding
Swift type string representations, which are used in schema generation for
structured output and tool definitions.
"""

from typing import Type, Union


def _python_type_to_string(python_type: Type) -> str:
    """Convert a Python type to a Swift type string representation for the schema.

    This function maps Python types to their corresponding Swift type strings
    that are used in the Foundation Models schema system. It handles basic types
    (str, int, float, bool), generic list types with element specifications,
    and Optional types (Union[T, None]).

    :param python_type: The Python type to convert. Can be a basic type (str, int,
                        float, bool), a generic type like List[T], or Optional[T].
    :type python_type: Type

    :return: A string representation of the corresponding Swift type. Basic types
             are mapped to their JSON schema equivalents (for example, "string", "integer",
             "number", "boolean"). List types are represented as "array<element_type>".
             For unrecognized types, returns the class name.
    :rtype: str

    :raises TypeError: If a generic list type is used without specifying an element
                       type (for example, using `list` instead of `List[str]`).

    Example:
        >>> _python_type_to_string(str)
        'string'
        >>> _python_type_to_string(int)
        'integer'
        >>> from typing import List
        >>> _python_type_to_string(List[str])
        'array<string>'
        >>> from typing import Optional
        >>> _python_type_to_string(Optional[int])
        'integer'

    .. note::
       Generic list types must always specify an element type. Using the bare
       `list` type will raise a TypeError.

    .. seealso::
       This function is used internally by the schema generation system to create
       type specifications for structured output and tool parameters.
    """
    if python_type is str:
        return "string"
    elif python_type is int:
        return "integer"
    elif python_type is float:
        return "number"
    elif python_type is bool:
        return "boolean"
    elif python_type is list:  # No generic lists allowed
        raise TypeError(
            "Generic list types must specify an element type, for example, List[str]"
        )
    elif hasattr(python_type, "__origin__"):
        if python_type.__origin__ is list:
            element_type = python_type.__args__[0] if python_type.__args__ else str
            element_type_str = _python_type_to_string(element_type)
            return f"array<{element_type_str}>"
        elif python_type.__origin__ is Union:
            # Handle Optional[T] -> Union[T, None]
            non_none_types = [
                arg for arg in python_type.__args__ if arg is not type(None)
            ]
            if len(non_none_types) == 1:
                return _python_type_to_string(non_none_types[0])

    # Default to the class name
    return getattr(python_type, "__name__", str(python_type))
