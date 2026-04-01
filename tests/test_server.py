import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from apple_fm_cli.server import (
    APPLE_FM_CODEX_INSTRUCTIONS,
    app,
    build_responses_prompt,
)

client = TestClient(app)


def parse_sse_events(body: str) -> list[tuple[str | None, str]]:
    events: list[tuple[str | None, str]] = []
    current_event = None
    current_data: list[str] = []

    for line in body.splitlines():
        if not line:
            if current_event is not None or current_data:
                events.append((current_event, "\n".join(current_data)))
            current_event = None
            current_data = []
            continue
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
            continue
        if line.startswith("data: "):
            current_data.append(line.removeprefix("data: "))

    if current_event is not None or current_data:
        events.append((current_event, "\n".join(current_data)))

    return events


def event_names(body: str) -> list[str | None]:
    return [event for event, _ in parse_sse_events(body)]


def last_session_instructions(mock_fm_sdk):
    return mock_fm_sdk["session_ctor"].call_args.kwargs["instructions"]


@pytest.fixture
def mock_fm_sdk():
    with (
        patch("apple_fm_sdk.SystemLanguageModel") as mock_model,
        patch("apple_fm_sdk.LanguageModelSession") as mock_session,
    ):
        mock_model_inst = MagicMock()
        mock_model_inst.is_available.return_value = (True, "")
        mock_model.return_value = mock_model_inst

        mock_session_inst = MagicMock()
        mock_session_inst.respond = AsyncMock()
        mock_session_inst.token_usage = AsyncMock(
            return_value={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        mock_session.return_value = mock_session_inst

        yield {
            "model": mock_model_inst,
            "session": mock_session_inst,
            "session_ctor": mock_session,
        }


def test_chat_completions_basic(mock_fm_sdk):
    # Given
    mock_fm_sdk["session"].respond.return_value = MagicMock(text="Hello world")

    # When
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "model": "apple-fm"},
    )

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello world"
    assert data["object"] == "chat.completion"
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


def test_chat_completions_streaming(mock_fm_sdk):
    # Given
    async def mock_stream(prompt):
        yield "Hello "
        yield "Hello world"

    mock_fm_sdk["session"].stream_response = mock_stream

    # When
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "model": "apple-fm", "stream": True},
    )

    # Then
    assert response.status_code == 200
    assert response.text.count("Hello ") == 1
    assert response.text.count("world") == 1
    assert response.text.rstrip().endswith("data: [DONE]")


def test_chat_completions_json_schema(mock_fm_sdk):
    # Given
    from dataclasses import make_dataclass

    Cat = make_dataclass("Cat", [("name", str), ("age", int)])
    mock_fm_sdk["session"].respond.return_value = Cat(name="Whiskers", age=3)

    # When
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Generate a cat"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Cat",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    },
                },
            },
        },
    )

    # Then
    assert response.status_code == 200
    data = response.json()
    content = json.loads(data["choices"][0]["message"]["content"])
    assert content["name"] == "Whiskers"
    assert content["age"] == 3


def test_responses_api_basic(mock_fm_sdk):
    # Given
    mock_fm_sdk["session"].respond.return_value = MagicMock(text="2 + 2 equals 4.")
    mock_fm_sdk["session"].token_usage = AsyncMock(
        return_value={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )

    # When
    response = client.post(
        "/v1/responses",
        json={"model": "apple-fm", "input": "What is 2 + 2?", "instructions": "Math tutor"},
    )

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["id"].startswith("resp_")
    assert data["output"][0]["content"][0]["text"] == "2 + 2 equals 4."
    assert data["usage"]["total_tokens"] == 15


def test_responses_api_streaming(mock_fm_sdk):
    # Given
    async def mock_stream(prompt):
        yield "The answer "
        yield "The answer is 4."

    mock_fm_sdk["session"].stream_response = mock_stream

    # When
    response = client.post(
        "/v1/responses", json={"model": "apple-fm", "input": "What is 2 + 2?", "stream": True}
    )

    # Then
    assert response.status_code == 200
    payloads = [json.loads(data) for _, data in parse_sse_events(response.text) if data != "[DONE]"]
    deltas = [
        payload["delta"]
        for payload in payloads
        if payload.get("type") == "response.output_text.delta"
    ]
    assert deltas == ["The answer ", "is 4."]
    assert event_names(response.text) == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
        None,
    ]

    completed_event = parse_sse_events(response.text)[-2]
    assert completed_event[0] == "response.completed"
    assert json.loads(completed_event[1])["type"] == "response.completed"


def test_responses_api_streaming_deduplicates_multiline_snapshots(mock_fm_sdk):
    # Given
    async def mock_stream(prompt):
        yield "Step 1\n"
        yield "Step 1\nStep 2\n"
        yield "Step 1\nStep 2\nStep 3"

    mock_fm_sdk["session"].stream_response = mock_stream

    # When
    response = client.post(
        "/v1/responses",
        json={
            "model": "apple-fm",
            "input": "Give me three steps",
            "stream": True,
        },
    )

    # Then
    assert response.status_code == 200
    payloads = [json.loads(data) for _, data in parse_sse_events(response.text) if data != "[DONE]"]
    deltas = [
        payload["delta"]
        for payload in payloads
        if payload.get("type") == "response.output_text.delta"
    ]
    assert deltas == ["Step 1\n", "Step 2\n", "Step 3"]


def test_responses_api_codex_interactive_streaming_keeps_deltas_and_includes_final_text_content(
    mock_fm_sdk,
):
    # Given
    async def mock_stream(prompt):
        yield "2"

    mock_fm_sdk["session"].stream_response = mock_stream
    codex_instructions = (
        "You are a coding agent running in the Codex CLI, a terminal-based coding assistant.\n"
        "Codex CLI is an open source project led by OpenAI."
    )

    # When
    response = client.post(
        "/v1/responses",
        json={
            "model": "fm",
            "input": "what is 1 + 1",
            "instructions": codex_instructions,
            "stream": True,
        },
    )

    # Then
    assert response.status_code == 200
    payloads = [json.loads(data) for _, data in parse_sse_events(response.text) if data != "[DONE]"]
    deltas = [
        payload["delta"]
        for payload in payloads
        if payload.get("type") == "response.output_text.delta"
    ]
    assert deltas == ["2"]
    item_done = next(
        payload for payload in payloads if payload.get("type") == "response.output_item.done"
    )
    assert item_done["item"]["content"] == [{"type": "output_text", "text": "2"}]


def test_responses_api_codex_exec_streaming_uses_final_item_content(mock_fm_sdk):
    # Given
    async def mock_stream(prompt):
        yield "2"

    mock_fm_sdk["session"].stream_response = mock_stream
    codex_instructions = (
        "You are a coding agent running in the Codex CLI, a terminal-based coding assistant.\n"
        "Codex CLI is an open source project led by OpenAI."
    )

    # When
    response = client.post(
        "/v1/responses",
        json={
            "model": "fm",
            "input": "what is 1 + 1",
            "instructions": codex_instructions,
            "stream": True,
        },
        headers={"user-agent": "codex_exec/0.116.0"},
    )

    # Then
    assert response.status_code == 200
    payloads = [json.loads(data) for _, data in parse_sse_events(response.text) if data != "[DONE]"]
    assert not any(payload.get("type") == "response.output_text.delta" for payload in payloads)
    item_done = next(
        payload for payload in payloads if payload.get("type") == "response.output_item.done"
    )
    assert item_done["item"]["content"] == [{"type": "output_text", "text": "2"}]
    completed = next(payload for payload in payloads if payload.get("type") == "response.completed")
    assert completed["response"]["output"] == []


def test_responses_api_json_schema(mock_fm_sdk):
    # Given
    from dataclasses import make_dataclass

    Cat = make_dataclass("Cat", [("name", str), ("age", int)])
    mock_fm_sdk["session"].respond.return_value = Cat(name="Whiskers", age=3)
    mock_fm_sdk["session"].token_usage = AsyncMock(
        return_value={"total_tokens": 20, "prompt_tokens": 10, "completion_tokens": 10}
    )

    # When
    response = client.post(
        "/v1/responses",
        json={
            "model": "apple-fm",
            "input": "Generate a cat",
            "text": {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Cat",
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                        },
                    },
                }
            },
        },
    )

    # Then
    assert response.status_code == 200
    data = response.json()
    content = json.loads(data["output"][0]["content"][0]["text"])
    assert content["name"] == "Whiskers"
    assert content["age"] == 3


def test_responses_api_adapts_codex_instructions(mock_fm_sdk):
    # Given
    mock_fm_sdk["session"].respond.return_value = MagicMock(text="2")
    codex_instructions = (
        "You are a coding agent running in the Codex CLI, a terminal-based coding assistant.\n"
        "Codex CLI is an open source project led by OpenAI."
    )

    # When
    response = client.post(
        "/v1/responses",
        json={
            "model": "fm",
            "input": "what is 1 + 1",
            "instructions": codex_instructions,
        },
    )

    # Then
    assert response.status_code == 200
    assert last_session_instructions(mock_fm_sdk) == APPLE_FM_CODEX_INSTRUCTIONS
    assert mock_fm_sdk["session_ctor"].call_args.kwargs["model"] is not None


def test_build_responses_prompt_preserves_codex_conversation_history():
    # Given
    input_data = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "<environment_context>\nfoo\n</environment_context>"}
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "provide a recipe for chocolate cake"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Here is a chocolate cake recipe."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What cake are we making?"}],
        },
    ]

    # When
    prompt = build_responses_prompt(input_data, codex_mode=True)

    # Then
    assert prompt == (
        "User: provide a recipe for chocolate cake\n\n"
        "Assistant: Here is a chocolate cake recipe.\n\n"
        "User: What cake are we making?"
    )


def test_responses_api_preserves_regular_instructions(mock_fm_sdk):
    # Given
    mock_fm_sdk["session"].respond.return_value = MagicMock(text="2 + 2 equals 4.")

    # When
    response = client.post(
        "/v1/responses",
        json={
            "model": "fm",
            "input": "What is 2 + 2?",
            "instructions": "Math tutor",
        },
    )

    # Then
    assert response.status_code == 200
    assert last_session_instructions(mock_fm_sdk) == "Math tutor"
    assert mock_fm_sdk["session_ctor"].call_args.kwargs["model"] is None


def test_build_responses_prompt_strips_environment_context_for_codex_mode():
    # Given
    input_data = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "<environment_context>\nfoo\n</environment_context>"}
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "what is 1 + 1"}],
        },
    ]

    # When
    prompt = build_responses_prompt(input_data, codex_mode=True)

    # Then
    assert prompt == "User: what is 1 + 1"
