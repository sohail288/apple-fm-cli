# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import base64
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ContentPartType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class ImageContentPart:
    """Represents an image input for multimodal inference."""

    data: bytes
    format: str = "png"  # e.g., "png", "jpeg"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "image",
            "image": {"data": base64.b64encode(self.data).decode("utf-8"), "format": self.format},
        }


@dataclass
class AudioContentPart:
    """Represents an audio input for multimodal inference."""

    data: bytes
    format: str = "wav"  # e.g., "wav", "mp3"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "audio",
            "audio": {"data": base64.b64encode(self.data).decode("utf-8"), "format": self.format},
        }


ContentPart = str | ImageContentPart | AudioContentPart
