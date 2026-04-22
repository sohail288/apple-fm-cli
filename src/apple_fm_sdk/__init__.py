# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Foundation Models SDK for Python package initialization.
"""

from .content import (
    AudioContentPart,
    ContentPart,
    ImageContentPart,
)
from .core import (
    SystemLanguageModel,
    SystemLanguageModelGuardrails,
    SystemLanguageModelUnavailableReason,
    SystemLanguageModelUseCase,
    get_sentence_embedding,
)
from .errors import (
    AssetsUnavailableError,
    ConcurrentRequestsError,
    DecodingFailureError,
    ExceededContextWindowSizeError,
    FoundationModelsError,
    GenerationError,
    GenerationErrorCode,
    GuardrailViolationError,
    InvalidGenerationSchemaError,
    RateLimitedError,
    RefusalError,
    ToolCallError,
    UnsupportedGuideError,
    UnsupportedLanguageOrLocaleError,
)
from .generable import (
    ConvertibleFromGeneratedContent,
    ConvertibleToGeneratedContent,
    Generable,
    GeneratedContent,
    GenerationID,
)
from .generable_utils import generable
from .generation_guide import GenerationGuide, GuideType, guide
from .generation_options import GenerationOptions, SamplingMode, SamplingModeType
from .generation_schema import GenerationSchema
from .session import LanguageModelSession
from .tokenizer import Tokenizer
from .tool import Tool
from .transcript import Transcript

__version__ = "0.2.0"
__all__ = [
    "SystemLanguageModel",
    "LanguageModelSession",
    "Transcript",
    "SystemLanguageModelUseCase",
    "SystemLanguageModelGuardrails",
    "SystemLanguageModelUnavailableReason",
    "get_sentence_embedding",
    "Tool",
    "FoundationModelsError",
    "GenerationError",
    "ExceededContextWindowSizeError",
    "AssetsUnavailableError",
    "GuardrailViolationError",
    "UnsupportedGuideError",
    "UnsupportedLanguageOrLocaleError",
    "InvalidGenerationSchemaError",
    "DecodingFailureError",
    "RateLimitedError",
    "ConcurrentRequestsError",
    "RefusalError",
    "ToolCallError",
    "GenerationErrorCode",
    "generable",
    "guide",
    "GenerationSchema",
    "GeneratedContent",
    "GenerationGuide",
    "GuideType",
    "GenerationOptions",
    "SamplingMode",
    "SamplingModeType",
    "GenerationID",
    "ConvertibleFromGeneratedContent",
    "ConvertibleToGeneratedContent",
    "Generable",
    "ImageContentPart",
    "AudioContentPart",
    "ContentPart",
    "Tokenizer",
]
