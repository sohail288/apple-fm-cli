# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
This module provides the core Python bindings for Foundation Models, including
system foundation model access and configuration.

The main classes provided are:

* :class:`SystemLanguageModel` - Interface to the on-device foundation model used by Apple Intelligence
* :class:`SystemLanguageModelUseCase` - Enumeration of model use cases
* :class:`SystemLanguageModelGuardrails` - Enumeration of guardrail settings
* :class:`SystemLanguageModelUnavailableReason` - Enumeration of unavailability reasons

Example:
    Basic usage of SystemLanguageModel::

        import apple_fm_sdk as fm

        model = fm.SystemLanguageModel()
        is_available, reason = model.is_available()
        if is_available:
            # Use the model
            pass
"""

from .c_helpers import (
    _ManagedObject,
)
from enum import IntEnum
from typing import Optional
from .errors import FoundationModelsError

import ctypes
from ctypes import c_int

try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise ImportError(
        "Foundation Models C bindings not found. Please ensure _foundationmodels_ctypes.py is available."
    )


class SystemLanguageModelUnavailableReason(IntEnum):
    """Reasons why a system foundation model might be unavailable.

    This enumeration defines the possible reasons why a system foundation model
    cannot be used on the current device or in the current context.

    Attributes:
        APPLE_INTELLIGENCE_NOT_ENABLED: Apple Intelligence features are not enabled
            on this device or for this user.
        DEVICE_NOT_ELIGIBLE: The device does not meet the requirements for running
            the system language model.
        MODEL_NOT_READY: The model is still being downloaded or prepared and is not
            yet ready for use.
        UNKNOWN: The reason for unavailability is unknown or not specified.
    """

    APPLE_INTELLIGENCE_NOT_ENABLED = 0
    DEVICE_NOT_ELIGIBLE = 1
    MODEL_NOT_READY = 2
    UNKNOWN = 0xFF


class SystemLanguageModelUseCase(IntEnum):
    """Use cases for system foundation models.

    This enumeration defines the different use cases that can be specified when
    creating a system foundation model. The use case helps optimize the model's
    behavior for specific tasks.

    Attributes:
        GENERAL: General-purpose foundation model use case suitable for a wide range
            of natural language processing tasks.
        CONTENT_TAGGING: Specialized use case optimized for content classification
            and tagging tasks.
    """

    GENERAL = 0
    CONTENT_TAGGING = 1


class SystemLanguageModelGuardrails(IntEnum):
    """Guardrail settings for system foundation models.

    This enumeration defines the different levels of content filtering and safety
    guardrails that can be applied to system foundation models. Guardrails help
    ensure appropriate and safe model behavior.

    Attributes:
        DEFAULT: Standard guardrails with balanced content filtering appropriate
            for general use cases.
        PERMISSIVE_CONTENT_TRANSFORMATIONS: More permissive guardrails that allow
            greater flexibility in content transformation tasks while maintaining
            basic safety measures.
    """

    DEFAULT = 0
    PERMISSIVE_CONTENT_TRANSFORMATIONS = 1


class SystemLanguageModel(_ManagedObject):
    """Represents the on-device foundation model used by Apple Intelligence.

    This class provides the main interface for interacting with the system foundation
    model. It manages the lifecycle of the underlying C model object and provides
    methods to check availability and configure model behavior.
    """

    def __init__(
        self,
        use_case: SystemLanguageModelUseCase = SystemLanguageModelUseCase.GENERAL,
        guardrails: SystemLanguageModelGuardrails = SystemLanguageModelGuardrails.DEFAULT,
        _ptr=None,
    ):
        if _ptr is not None:
            super().__init__(_ptr)
        else:
            ptr = lib.FMSystemLanguageModelCreate(use_case.value, guardrails.value)
            super().__init__(ptr)

    def is_available(
        self,
    ) -> tuple[bool, Optional[SystemLanguageModelUnavailableReason]]:
        reason = c_int()
        is_available = lib.FMSystemLanguageModelIsAvailable(
            self._ptr, ctypes.byref(reason)
        )

        if is_available:
            return True, None
        else:
            return False, SystemLanguageModelUnavailableReason(reason.value)

    def token_count(self, text: str) -> int:
        """Measure how many tokens a string uses.

        This method uses the native FoundationModels framework to count tokens.
        On systems where the native tokenCount API is unavailable (pre-macOS 26.4),
        it returns -1.

        :param text: The text to count tokens for
        :type text: str
        :return: The number of tokens, or -1 if the native API is unavailable
        :rtype: int
        """
        count = lib.FMSystemLanguageModelGetTokenCount(self._ptr, text.encode("utf-8"))
        return int(count)

    @property
    def context_size(self) -> int:
        """Get the maximum context size — in tokens — that the model supports.

        This property returns the native context size reported by the framework.

        :return: The maximum context size in tokens
        :rtype: int
        """
        return int(lib.FMSystemLanguageModelGetContextSize(self._ptr))
