# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import ctypes
from enum import Enum
from typing import Any

try:
    from . import _ctypes_bindings as lib
except ImportError as e:
    raise ImportError("Python C bindings missing") from e


class GuideType(Enum):
    """Enumeration of available generation guide types.

    This enum defines the different types of constraints that can be applied
    to guide model generation behavior.

    :cvar any_of: Constraints output to one of a list of strings
    :cvar constant: Enforces a fixed string value
    :cvar count: Specifies the required number of items in an array
    :cvar element: Applies a guide to elements within an array
    :cvar max_items: Sets a maximum count for items in an array
    :cvar maximum: Sets a maximum numeric value constraint
    :cvar min_items: Sets a minimum count for items in an array
    :cvar minimum: Sets a minimum numeric value constraint
    :cvar range: Sets both minimum and maximum numeric constraints
    :cvar regex: Enforces a regex pattern constraint on strings
    """

    any_of = "enum"  # Serializes to "enum" in JSON schema
    constant = "constant"  # Represented by enum of 1 value
    count = "count"
    element = "element"  # Enforces a guide on the elements within the array.
    max_items = "maxItems"  # called maximumCount in Swift
    maximum = "maximum"
    min_items = "minItems"  # called minimumCount in Swift
    minimum = "minimum"
    range = "range"
    regex = "regex"  # Enforces a limited regex vocabulary -> serializes to "pattern"


class GenerationGuide:
    """Represents a constraint used to guide the generation of a property.

    Guides are used in conjunction with GenerationSchema to enforce specific
    data formats, ranges, or choices during model generation.

    :param guide_type: The type of guide constraint
    :type guide_type: GuideType
    :param value: The value(s) for the constraint
    :type value: Any, optional
    """

    def __init__(self, guide_type: GuideType, value: Any = None) -> None:
        """Initialize a generation guide.

        :param guide_type: The type of guide constraint
        :type guide_type: GuideType
        :param value: The value(s) for the constraint
        :type value: Any, optional
        """
        self.guide_type = guide_type
        self.value = value

    @classmethod
    def any_of(cls, values: list[str]) -> GenerationGuide:
        """Create an any_of guide for strings.

        Constrains the output to be one of the specified string values.

        :param values: List of valid string choices
        :type values: List[str]
        :return: A GenerationGuide with any_of constraint
        :rtype: GenerationGuide

        Example::

            guide = GenerationGuide.any_of(["red", "green", "blue"])
        """
        return cls(GuideType.any_of, values)

    @classmethod
    def constant(cls, value: str) -> GenerationGuide:
        """Enforce that the string be precisely the given value.

        :param value: The exact string value required
        :type value: str
        :return: A GenerationGuide with constant constraint
        :rtype: GenerationGuide
        """
        return cls(GuideType.constant, value)

    @classmethod
    def count(cls, value: int) -> GenerationGuide:
        """Specify exactly how many elements should be in an array.

        :param value: The required number of array elements
        :type value: int
        :return: A GenerationGuide with count constraint
        :rtype: GenerationGuide
        """
        return cls(GuideType.count, value)

    @classmethod
    def element(cls, guide: GenerationGuide) -> GenerationGuide:
        """Apply a guide to each element within an array.

        :param guide: The guide to apply to array elements
        :type guide: GenerationGuide
        :return: A GenerationGuide with element constraint
        :rtype: GenerationGuide
        """
        return cls(GuideType.element, guide)

    @classmethod
    def max_items(cls, value: int) -> GenerationGuide:
        """Set a maximum number of elements in an array.

        :param value: The maximum allowed number of elements
        :type value: int
        :return: A GenerationGuide with max_items constraint
        :rtype: GenerationGuide
        """
        return cls(GuideType.max_items, value)

    @classmethod
    def maximum(cls, value: float) -> GenerationGuide:
        """Set a maximum numeric value.

        :param value: The maximum allowed numeric value
        :type value: float
        :return: A GenerationGuide with maximum constraint
        :rtype: GenerationGuide
        """
        return cls(GuideType.maximum, value)

    @classmethod
    def min_items(cls, value: int) -> GenerationGuide:
        """Set a minimum number of elements in an array.

        :param value: The minimum allowed number of elements
        :type value: int
        :return: A GenerationGuide with min_items constraint
        :rtype: GenerationGuide
        """
        return cls(GuideType.min_items, value)

    @classmethod
    def minimum(cls, value: float) -> GenerationGuide:
        """Set a minimum numeric value.

        :param value: The minimum allowed numeric value
        :type value: float
        :return: A GenerationGuide with minimum constraint
        :rtype: GenerationGuide
        """
        return cls(GuideType.minimum, value)

    @classmethod
    def range(cls, min_val: float, max_val: float) -> GenerationGuide:
        """Set both minimum and maximum numeric constraints.

        :param min_val: The minimum allowed numeric value
        :type min_val: float
        :param max_val: The maximum allowed numeric value
        :type max_val: float
        :return: A GenerationGuide with range constraint
        :rtype: GenerationGuide
        """
        return cls(GuideType.range, (min_val, max_val))

    @classmethod
    def regex(cls, pattern: str) -> GenerationGuide:
        """Enforce a regex pattern constraint on strings.

        :param pattern: The regex pattern to enforce
        :type pattern: str
        :return: A GenerationGuide with regex constraint
        :rtype: GenerationGuide

        .. note::
            The model supports a subset of regex vocabulary optimized for
            on-device generation.
        """
        return cls(GuideType.regex, pattern)

    def _apply_to_c_property(self, prop_ptr: Any, wrapped: bool = False) -> None:
        """Apply this guide to a C property object.

        Internal method used to bridge Python guides to the C library.

        :param prop_ptr: Pointer to the C property object
        :type prop_ptr: Any
        :param wrapped: Whether this guide is being applied as a nested
            element guide
        :type wrapped: bool, optional
        """
        guide_type = self.guide_type
        value = self.value

        if guide_type == GuideType.element:
            guide_type = self.value.guide_type
            value = self.value.value
            wrapped = True

        # Handle guide types that have direct C bindings
        if guide_type == GuideType.any_of:
            self.convert_any_of_to_c(any_of=value, prop_ptr=prop_ptr, wrapped=wrapped)
        elif guide_type == GuideType.constant:
            self.convert_any_of_to_c(any_of=[value], prop_ptr=prop_ptr, wrapped=wrapped)
        elif guide_type == GuideType.count:
            lib.FMGenerationSchemaPropertyAddCountGuide(prop_ptr, int(value), wrapped)
        elif guide_type == GuideType.max_items:
            lib.FMGenerationSchemaPropertyAddMaxItemsGuide(prop_ptr, int(value), wrapped)
        elif guide_type == GuideType.maximum:
            lib.FMGenerationSchemaPropertyAddMaximumGuide(prop_ptr, float(value), wrapped)
        elif guide_type == GuideType.min_items:
            lib.FMGenerationSchemaPropertyAddMinItemsGuide(prop_ptr, int(value), wrapped)
        elif guide_type == GuideType.minimum:
            lib.FMGenerationSchemaPropertyAddMinimumGuide(prop_ptr, float(value), wrapped)
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

    def convert_any_of_to_c(self, any_of: list[str], prop_ptr: Any, wrapped: bool = False) -> None:
        """Convert any_of constraint to C library calls.

        Handles the conversion of string choice constraints to C data structures
        and invokes the appropriate C library function.

        :param any_of: List of valid string choices
        :type any_of: List[str]
        :param prop_ptr: Pointer to the C property object
        :type prop_ptr: Any
        :param wrapped: Whether this guide is wrapped (for element guides)
        :type wrapped: bool, optional

        .. note::
            This method creates C string buffers and pointer arrays to pass
            the string choices to the C library. Memory management is handled
            automatically through Python's garbage collection.
        """
        c_strings = [ctypes.create_string_buffer(c.encode("utf-8")) for c in any_of]
        choice_ptrs = (ctypes.POINTER(ctypes.c_char) * len(any_of))(
            *[ctypes.cast(s, ctypes.POINTER(ctypes.c_char)) for s in c_strings]
        )
        lib.FMGenerationSchemaPropertyAddAnyOfGuide(prop_ptr, choice_ptrs, len(any_of), wrapped)


def guide(
    description: str | None = None,
    *,
    any_of: list[str] | None = None,
    constant: str | None = None,
    count: int | None = None,
    element: GenerationGuide | None = None,
    max_items: int | None = None,
    maximum: float | None = None,
    min_items: int | None = None,
    minimum: float | None = None,
    range: tuple[float, float] | None = None,
    regex: str | None = None,
) -> GenerationGuide | None:
    """Helper function to create a GenerationGuide from keyword arguments.

    This function provides a convenient way to define guides when creating
    property definitions.

    :param description: Optional description for the guide
    :type description: str, optional
    :param any_of: Constraints output to one of the provided strings
    :type any_of: List[str], optional
    :param constant: Enforces a fixed string value
    :type constant: str, optional
    :param count: Specifies the required number of items in an array
    :type count: int, optional
    :param element: Applies a guide to elements within an array
    :type element: GenerationGuide, optional
    :param max_items: Sets a maximum count for items in an array
    :type max_items: int, optional
    :param maximum: Sets a maximum numeric value constraint
    :type maximum: float, optional
    :param min_items: Sets a minimum count for items in an array
    :type min_items: int, optional
    :param minimum: Sets a minimum numeric value constraint
    :type minimum: float, optional
    :param range: Sets both minimum and maximum numeric constraints
    :type range: tuple[float, float], optional
    :param regex: Enforces a regex pattern constraint on strings
    :type regex: str, optional
    :return: A GenerationGuide instance if any constraint was provided, else None
    :rtype: GenerationGuide, optional
    """
    if any_of is not None:
        return GenerationGuide.any_of(any_of)
    if constant is not None:
        return GenerationGuide.constant(constant)
    if count is not None:
        return GenerationGuide.count(count)
    if element is not None:
        return GenerationGuide.element(element)
    if max_items is not None:
        return GenerationGuide.max_items(max_items)
    if maximum is not None:
        return GenerationGuide.maximum(maximum)
    if min_items is not None:
        return GenerationGuide.min_items(min_items)
    if minimum is not None:
        return GenerationGuide.minimum(minimum)
    if range is not None:
        return GenerationGuide.range(*range)
    if regex is not None:
        return GenerationGuide.regex(regex)
    return None
