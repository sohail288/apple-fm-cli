# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Foundation Models error classes and exception types.
"""

from enum import IntEnum
from typing import Optional


class FoundationModelsError(Exception):
    """Base exception for Foundation Models errors."""

    pass


class GenerationError(FoundationModelsError):
    """Base class for generation-specific errors."""

    pass


class InvalidGenerationSchemaError(FoundationModelsError):
    """Error thrown when a generation schema is invalid."""

    pass


class ExceededContextWindowSizeError(GenerationError):
    """Error thrown when the context window size is exceeded."""

    pass


class AssetsUnavailableError(GenerationError):
    """Error thrown when required assets are unavailable."""

    pass


class GuardrailViolationError(GenerationError):
    """Error thrown when a guardrail violation occurs."""

    pass


class UnsupportedGuideError(GenerationError):
    """Error thrown when an unsupported guide is used."""

    pass


class UnsupportedLanguageOrLocaleError(GenerationError):
    """Error thrown when an unsupported language or locale is used."""

    pass


class DecodingFailureError(GenerationError):
    """Error thrown when decoding fails."""

    pass


class RateLimitedError(GenerationError):
    """Error thrown when rate limited."""

    pass


class ConcurrentRequestsError(GenerationError):
    """Error thrown when there are too many concurrent requests."""

    pass


class RefusalError(GenerationError):
    """Error thrown when the model refuses to generate content."""

    def __init__(
        self,
        message: str,
        debug_description: Optional[str] = None,
        explanation_entries=None,
    ):
        super().__init__(message, debug_description)
        self.explanation_entries = explanation_entries or []


class ToolCallError(FoundationModelsError):
    """Error thrown when a tool call fails."""

    def __init__(self, tool_name: str, underlying_error: Exception):
        super().__init__(f"Tool '{tool_name}' failed: {underlying_error}")
        self.tool_name = tool_name
        self.underlying_error = underlying_error


class GenerationErrorCode(IntEnum):
    """Error codes for generation errors that map to C status codes."""

    SUCCESS = 0
    EXCEEDED_CONTEXT_WINDOW_SIZE = 1
    ASSETS_UNAVAILABLE = 2
    GUARDRAIL_VIOLATION = 3
    UNSUPPORTED_GUIDE = 4
    UNSUPPORTED_LANGUAGE_OR_LOCALE = 5
    DECODING_FAILURE = 6
    RATE_LIMITED = 7
    CONCURRENT_REQUESTS = 8
    REFUSAL = 9
    INVALID_SCHEMA = 10
    UNKNOWN_ERROR = 255


def _status_code_to_exception(
    status_code: int, debug_description: Optional[str] = None
) -> GenerationError:
    """Convert a C status code to the appropriate GenerationError subclass."""
    error_map = {
        GenerationErrorCode.EXCEEDED_CONTEXT_WINDOW_SIZE: ExceededContextWindowSizeError,
        GenerationErrorCode.ASSETS_UNAVAILABLE: AssetsUnavailableError,
        GenerationErrorCode.GUARDRAIL_VIOLATION: GuardrailViolationError,
        GenerationErrorCode.UNSUPPORTED_GUIDE: UnsupportedGuideError,
        GenerationErrorCode.UNSUPPORTED_LANGUAGE_OR_LOCALE: UnsupportedLanguageOrLocaleError,
        GenerationErrorCode.DECODING_FAILURE: DecodingFailureError,
        GenerationErrorCode.RATE_LIMITED: RateLimitedError,
        GenerationErrorCode.CONCURRENT_REQUESTS: ConcurrentRequestsError,
        GenerationErrorCode.REFUSAL: RefusalError,
        GenerationErrorCode.INVALID_SCHEMA: InvalidGenerationSchemaError,
    }

    error_messages = {
        GenerationErrorCode.EXCEEDED_CONTEXT_WINDOW_SIZE: "Context window size exceeded",
        GenerationErrorCode.ASSETS_UNAVAILABLE: "Required assets are unavailable",
        GenerationErrorCode.GUARDRAIL_VIOLATION: "Guardrail violation occurred",
        GenerationErrorCode.UNSUPPORTED_GUIDE: "Unsupported guide used",
        GenerationErrorCode.UNSUPPORTED_LANGUAGE_OR_LOCALE: "Unsupported language or locale",
        GenerationErrorCode.DECODING_FAILURE: "Failed to decode response",
        GenerationErrorCode.RATE_LIMITED: "Request was rate limited",
        GenerationErrorCode.CONCURRENT_REQUESTS: "Too many concurrent requests",
        GenerationErrorCode.REFUSAL: "Model refused to generate content",
        GenerationErrorCode.INVALID_SCHEMA: "Invalid generation schema provided",
    }

    try:
        error_code = GenerationErrorCode(status_code)
    except ValueError:
        # Unknown error code
        return GenerationError(
            f"Unknown generation error (status: {status_code}): {debug_description}",
        )

    if error_code in error_map:
        error_class = error_map[error_code]
        message = error_messages[error_code]
        return error_class(f"{message}: {debug_description}")
    else:
        return GenerationError(
            f"Generation error (status: {status_code}): {debug_description}"
        )
