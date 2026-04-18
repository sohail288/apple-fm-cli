from typing import Any, cast

import pytest

try:
    import apple_fm_sdk as fm
except ImportError as error:
    pytest.skip(
        f"Foundation Models C bindings unavailable: {error}",
        allow_module_level=True,
    )

if not hasattr(fm.LanguageModelSession, "token_usage"):
    pytest.skip(
        "LanguageModelSession token_usage is unavailable in the test fallback stub",
        allow_module_level=True,
    )


class FakeTranscript:
    async def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": {
                "entries": [
                    {
                        "role": "instructions",
                        "contents": [{"type": "text", "text": "You are a helper."}],
                    },
                    {
                        "role": "user",
                        "contents": [{"type": "text", "text": "What is 2 + 2?"}],
                    },
                    {
                        "role": "response",
                        "contents": [{"type": "text", "text": "2 + 2 equals 4."}],
                    },
                ]
            }
        }


@pytest.mark.asyncio
async def test_token_usage_falls_back_to_heuristic_when_native_count_is_unavailable() -> None:
    session = cast(Any, fm.LanguageModelSession.__new__(fm.LanguageModelSession))
    session.transcript = FakeTranscript()
    session.token_count = lambda text: -1

    usage = await session.token_usage()

    assert usage["instructions_tokens"] > 0
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == (
        usage["instructions_tokens"] + usage["prompt_tokens"] + usage["completion_tokens"]
    )
