# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
This module provides classes and utilities for creating generation guides that constrain
how foundation models generate text, similar to Swift's GenerationGuide functionality.

The main components are:

* :class:`GuideType`: Enumeration of available constraint types
* :class:`GenerationGuide`: Class representing a generation constraint
* :func:`guide`: Convenience function for creating guided dataclass fields

Example:
        import apple_fm_sdk as fm

        @fm.generable("A cat's profile")
        class Cat:
            name: str = guide("Cat's name")
            age: int = guide("Age in years", range=(0, 20))
            favoriteFood: str = guide("Favorite food", anyOf=["fish", "chicken", "tuna"])
"""

from typing import Any, List, Optional, Union
from dataclasses import field
from enum import Enum
import ctypes

try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise (ImportError("Python C bindings missing"))


class GuideType(Enum):
    """Enumeration of available generation guide types.

    This enum defines the different types of constraints that can be applied
    to guide model generation behavior.

    :cvar anyOf: Constrains output to a specific set of string values
    :cvar constant: Constrains output to a single constant value
    :cvar count: Sets exact number of items in collections
    :cvar element: Enforces guides on array elements
    :cvar maxItems: Sets maximum number of items in collections
    :cvar maximum: Sets a maximum numeric value constraint
    :cvar minItems: Sets minimum number of items in collections
    :cvar minimum: Sets a minimum numeric value constraint
    :cvar range: Sets both minimum and maximum numeric constraints
    :cvar regex: Enforces a regex pattern constraint on strings
    """

    anyOf = "enum"  # Serializes to "enum" in JSON schema
    constant = "constant"  # Represented by enum of 1 value
    count = "count"
    element = "element"  # Enforces a guide on the elements within the array.
    maxItems = "maxItems"  # called maximumCount in Swift
    maximum = "maximum"
    minItems = "minItems"  # called minimumCount in Swift
    minimum = "minimum"
    range = "range"
    regex = "regex"  # Enforces a limited regex vocabulary -> serializes to "pattern"


class GenerationGuide:
    """Represents constraints for generation, similar to Swift's GenerationGuide.

    This class encapsulates different types of constraints that can be applied
    to guide model generation, such as limiting choices, setting numeric ranges,
    or constraining collection sizes.

    :ivar guide_type: The type of constraint to apply
    :vartype guide_type: GuideType
    :ivar value: The constraint value(s) associated with the guide type
    :vartype value: Any
    """

    guide_type: GuideType
    value: Any

    def __init__(self, guide_type: GuideType, value: Any = None):
        """Initialize a GenerationGuide instance.

        :param guide_type: The type of guide constraint
        :type guide_type: GuideType
        :param value: The value(s) for the constraint
        :type value: Any, optional
        """
        self.guide_type = guide_type
        self.value = value

    @classmethod
    def anyOf(cls, values: List[str]) -> "GenerationGuide":
        """Create an anyOf guide for strings.

        Constrains the output to be one of the specified string values.

        :param values: List of valid string choices
        :type values: List[str]
        :return: A GenerationGuide with anyOf constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.anyOf(["red", "green", "blue"])
        """
        return cls(GuideType.anyOf, values)

    @classmethod
    def constant(cls, value: str) -> "GenerationGuide":
        """Enforce that the string be precisely the given value.

        :param value: The exact string value required
        :type value: str
        :return: A GenerationGuide with constant constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.constant("active")
        """
        return cls(GuideType.constant, value)

    @classmethod
    def count(cls, count: int) -> "GenerationGuide":
        """Enforce that the array has exactly a certain number of elements.

        :param count: The exact number of elements required
        :type count: int
        :return: A GenerationGuide with count constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.count(5)
        """
        return cls(GuideType.count, count)

    @classmethod
    def element(cls, guide: "GenerationGuide") -> "GenerationGuide":
        """Enforce a guide on the elements within the array.

        :param guide: The guide to apply to each array element
        :type guide: GenerationGuide
        :return: A GenerationGuide with element constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.element(GenerationGuide.range((0, 100)))
        """
        return cls(GuideType.element, guide)

    @classmethod
    def max_items(cls, value: int) -> "GenerationGuide":
        """Enforce a maximum number of elements in the array.

        :param value: Maximum number of elements allowed
        :type value: int
        :return: A GenerationGuide with max_items constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.max_items(10)
        """
        return cls(GuideType.maxItems, value)

    @classmethod
    def maximum(cls, value: Union[int, float]) -> "GenerationGuide":
        """Enforce a maximum value.

        :param value: Maximum numeric value allowed
        :type value: Union[int, float]
        :return: A GenerationGuide with maximum constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.maximum(100.0)
        """
        return cls(GuideType.maximum, value)

    @classmethod
    def min_items(cls, value: int) -> "GenerationGuide":
        """Enforce a minimum number of elements in the array.

        :param value: Minimum number of elements required
        :type value: int
        :return: A GenerationGuide with min_items constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.min_items(1)
        """
        return cls(GuideType.minItems, value)

    @classmethod
    def minimum(cls, value: Union[int, float]) -> "GenerationGuide":
        """Enforce a minimum value.

        :param value: Minimum numeric value allowed
        :type value: Union[int, float]
        :return: A GenerationGuide with minimum constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.minimum(0.0)
        """
        return cls(GuideType.minimum, value)

    @classmethod
    def range(cls, range_tuple: tuple) -> "GenerationGuide":
        """Enforce values fall within a range.

        :param range_tuple: Tuple of (min, max) values
        :type range_tuple: tuple
        :return: A GenerationGuide with range constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.range((0, 120))
        """
        return cls(GuideType.range, range_tuple)

    @classmethod
    def regex(cls, pattern: str) -> "GenerationGuide":
        """Enforce that the string matches the given regex pattern.

        :param pattern: Regular expression pattern to match
        :type pattern: str
        :return: A GenerationGuide with regex constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.regex(r"#/[a-zA-Z]+/#")
        """
        return cls(GuideType.regex, pattern)

    def convert_to_c(self, prop_ptr: Any):
        """Convert the generation guide to C library calls.

        Translates the Python generation guide into appropriate C library
        function calls to apply the constraints in the underlying foundation
        model system.

        :param prop_ptr: Pointer to the C property object that will receive
                        the guide constraints
        :type prop_ptr: Any
        :raises RuntimeError: If the guide_type is not supported or unknown

        .. note::
            This method handles the low-level conversion between Python objects
            and C data structures, including proper memory management for
            string arrays and type conversions for numeric values.

            Some guide types (minimum, maximum, minItems, maxItems, regex, element)
            are handled through serialization since they don't have direct C bindings.
        """
        guide_type = self.guide_type
        value = self.value
        wrapped = False  # Indicates if the guide is wrapped (for element guides)

        # Check for wrapped element guide
        if guide_type == GuideType.element:
            guide_type = self.value.guide_type
            value = self.value.value
            wrapped = True

        # Handle guide types that have direct C bindings
        if guide_type == GuideType.anyOf:
            self.convert_anyOf_to_c(anyOf=value, prop_ptr=prop_ptr, wrapped=wrapped)
        elif guide_type == GuideType.constant:
            # Constant is equivalent to anyOf with a single value
            self.convert_anyOf_to_c(anyOf=[value], prop_ptr=prop_ptr, wrapped=wrapped)
        elif guide_type == GuideType.count:
            lib.FMGenerationSchemaPropertyAddCountGuide(prop_ptr, int(value), wrapped)
        elif guide_type == GuideType.maxItems:
            lib.FMGenerationSchemaPropertyAddMaxItemsGuide(
                prop_ptr, int(value), wrapped
            )
        elif guide_type == GuideType.maximum:
            lib.FMGenerationSchemaPropertyAddMaximumGuide(
                prop_ptr, float(value), wrapped
            )
        elif guide_type == GuideType.minItems:
            lib.FMGenerationSchemaPropertyAddMinItemsGuide(
                prop_ptr, int(value), wrapped
            )
        elif guide_type == GuideType.minimum:
            lib.FMGenerationSchemaPropertyAddMinimumGuide(
                prop_ptr, float(value), wrapped
            )
        elif guide_type == GuideType.range:
            min_val, max_val = value
            lib.FMGenerationSchemaPropertyAddRangeGuide(
                prop_ptr, float(min_val), float(max_val), wrapped
            )
        elif guide_type == GuideType.regex:
            c_string = ctypes.create_string_buffer(value.encode("utf-8"))
            pattern_ptr = ctypes.cast(c_string, ctypes.POINTER(ctypes.c_char))
            lib.FMGenerationSchemaPropertyAddRegex(prop_ptr, pattern_ptr, wrapped)
        else:
            # Fallback for any unexpected guide types
            raise RuntimeError(f"Unknown or unsupported guide type: {self.guide_type}")

    def convert_anyOf_to_c(self, anyOf, prop_ptr: Any, wrapped: bool = False):
        """Convert anyOf constraint to C library calls.

        Handles the conversion of string choice constraints to C data structures
        and invokes the appropriate C library function.

        :param anyOf: List of valid string choices
        :type anyOf: List[str]
        :param prop_ptr: Pointer to the C property object
        :type prop_ptr: Any
        :param wrapped: Whether this guide is wrapped (for element guides)
        :type wrapped: bool, optional

        .. note::
            This method creates C string buffers and pointer arrays to pass
            the string choices to the C library. Memory management is handled
            automatically through Python's garbage collection.
        """
        c_strings = [ctypes.create_string_buffer(c.encode("utf-8")) for c in anyOf]
        choice_ptrs = (ctypes.POINTER(ctypes.c_char) * len(anyOf))(
            *[ctypes.cast(s, ctypes.POINTER(ctypes.c_char)) for s in c_strings]
        )
        lib.FMGenerationSchemaPropertyAddAnyOfGuide(
            prop_ptr, choice_ptrs, len(anyOf), wrapped
        )


def guide(
    description: Optional[str] = None,
    *,
    anyOf: Optional[List[str]] = None,
    constant: Optional[str] = None,
    count: Optional[int] = None,
    element: Optional["GenerationGuide"] = None,
    max_items: Optional[int] = None,
    maximum: Optional[Union[int, float]] = None,
    min_items: Optional[int] = None,
    minimum: Optional[Union[int, float]] = None,
    range: Optional[tuple] = None,
    regex: Optional[str] = None,
) -> Any:
    """Create a field with a guide, similar to Swift's @Guide annotation.

    This convenience function creates a dataclass field with generation guide
    metadata that can be used to constrain model output during generation.
    Multiple constraints can be applied to a single field.

    :param description: Description of the field for documentation purposes
    :type description: str, optional
    :param anyOf: List of valid string choices that constrain output to specific values
    :type anyOf: List[str], optional
    :param constant: Constrains output to a single constant value
    :type constant: str, optional
    :param count: Expected exact number of items for collections (must be positive)
    :type count: int, optional
    :param element: Guide to apply to array elements
    :type element: GenerationGuide, optional
    :param max_items: Maximum number of items for collections (must be non-negative)
    :type max_items: int, optional
    :param maximum: Maximum value for numeric types
    :type maximum: Union[int, float], optional
    :param min_items: Minimum number of items for collections (must be non-negative)
    :type min_items: int, optional
    :param minimum: Minimum value for numeric types
    :type minimum: Union[int, float], optional
    :param range: Tuple of (min, max) for numeric ranges
    :type range: tuple, optional
    :param regex: Regular expression pattern that the output must match
    :type regex: str, optional
    :return: A dataclass field with guide metadata attached
    :rtype: Any
    :raises ValueError: If any constraint values are invalid (for example, negative counts,
                       malformed ranges, non-string choices)

    Example:
        Basic field with description only::

            name: str = guide("The person's full name")

        Numeric field with range constraint::

            age: int = guide("Age in years", range=(0, 120))

        Collection with exact count::

            hobbies: List[str] = guide("List of hobbies", count=3)

        String field with choices::

            color: str = guide("Favorite color", anyOf=["red", "blue", "green"])

        Numeric field with separate min/max::

            score: float = guide("Test score", minimum=0.0, maximum=100.0)

        Collection with size constraints::

            tags: List[str] = guide("Tags list", min_items=1, max_items=5)

        Constant value constraint::

            status: str = guide("Status", constant="active")

        Regex pattern constraint::

            email: str = guide("Name", regex=r"#/[a-zA-Z]+/#")

        Element constraint for arrays::

            scores: List[int] = guide("Test scores", element=GenerationGuide.range((0, 100)))

    .. note::
        The guide function validates constraint parameters at creation time
        and stores them as metadata that can be processed by generation
        systems to enforce the specified constraints.
    """
    metadata: dict = {"description": description}
    guides = []

    if anyOf is not None:
        if not isinstance(anyOf, list) or not all(isinstance(c, str) for c in anyOf):
            raise ValueError("anyOf must be a list of strings")
        guides.append(GenerationGuide(GuideType.anyOf, anyOf))

    if constant is not None:
        if not isinstance(constant, str):
            raise ValueError("constant must be a string")
        guides.append(GenerationGuide(GuideType.constant, constant))

    if count is not None:
        if not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        guides.append(GenerationGuide(GuideType.count, count))

    if element is not None:
        if not isinstance(element, GenerationGuide):
            raise ValueError("element must be a GenerationGuide instance")
        guides.append(GenerationGuide(GuideType.element, element))

    if max_items is not None:
        if not isinstance(max_items, int) or max_items < 0:
            raise ValueError("max_items must be a non-negative integer")
        guides.append(GenerationGuide(GuideType.maxItems, max_items))

    if maximum is not None:
        if not isinstance(maximum, (int, float)):
            raise ValueError("maximum must be a number")
        guides.append(GenerationGuide(GuideType.maximum, maximum))

    if min_items is not None:
        if not isinstance(min_items, int) or min_items < 0:
            raise ValueError("min_items must be a non-negative integer")
        guides.append(GenerationGuide(GuideType.minItems, min_items))

    if minimum is not None:
        if not isinstance(minimum, (int, float)):
            raise ValueError("minimum must be a number")
        guides.append(GenerationGuide(GuideType.minimum, minimum))

    if range is not None:
        if not isinstance(range, tuple) or len(range) != 2:
            raise ValueError("range must be a tuple of (min, max)")
        guides.append(GenerationGuide(GuideType.range, range))

    if regex is not None:
        if not isinstance(regex, str):
            raise ValueError("regex must be a string")
        guides.append(GenerationGuide(GuideType.regex, regex))

    if guides:
        metadata["guides"] = guides

    return field(metadata=metadata)
