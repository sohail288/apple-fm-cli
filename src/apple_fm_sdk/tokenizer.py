# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .core import SystemLanguageModel


class Tokenizer:
    """Represents a tokenizer for an Apple Foundation Model.

    This class provides methods to interact with the model's tokenizer.
    Note: Direct encode/decode to token IDs is not yet exposed by the C bridge,
    but token counting is supported natively.
    """

    def __init__(self, model: Optional["SystemLanguageModel"] = None):
        from .core import SystemLanguageModel

        self._model = model if model else SystemLanguageModel()

    def encode(self, text: str) -> List[int]:
        """Convert a string into a sequence of token IDs.

        Note: This is currently not supported natively and will raise an error.
        """
        raise NotImplementedError("Native token ID encoding is not yet supported by the C bridge.")

    def decode(self, tokens: List[int]) -> str:
        """Convert a sequence of token IDs back into a string.

        Note: This is currently not supported natively and will raise an error.
        """
        raise NotImplementedError("Native token ID decoding is not yet supported by the C bridge.")

    def count(self, text: str) -> int:
        """Calculate the number of tokens in a string.

        :param text: The text to count
        :return: The token count
        """
        return self._model.token_count(text)
