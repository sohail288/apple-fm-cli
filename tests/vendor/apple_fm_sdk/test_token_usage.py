import pytest

import apple_fm_sdk as fm


@pytest.mark.asyncio
async def test_system_model_token_count():
    model = fm.SystemLanguageModel()
    count = model.token_count("Hello, world!")
    assert count > 0
    print(f"✓ Token count: {count}")


@pytest.mark.asyncio
async def test_session_token_usage():
    session = fm.LanguageModelSession(instructions="You are a helper.")
    await session.respond("What is 2 + 2?")

    usage = await session.token_usage()
    print(f"✓ Session usage: {usage}")

    assert usage["total_tokens"] > 0
    assert usage["instructions_tokens"] > 0
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0


def test_context_size():
    model = fm.SystemLanguageModel()
    assert model.context_size == 4096
    print(f"✓ Context size: {model.context_size}")
