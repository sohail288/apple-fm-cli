import base64
import dataclasses
import json
import struct
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from apple_fm_cli.server import APPLE_FM_CODEX_INSTRUCTIONS, build_responses_prompt, create_app


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


def last_session_instructions(mock_fm_sdk: dict[str, Any]) -> str | None:
    return cast(str | None, mock_fm_sdk["session_ctor"].call_args.kwargs["instructions"])


@pytest.fixture
def mock_fm_sdk() -> dict[str, Any]:
    class GenerationSchema:
        pass

    class SystemLanguageModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def is_available(self) -> tuple[bool, str]:
            return True, ""

    class SystemLanguageModelGuardrails:
        PERMISSIVE_CONTENT_TRANSFORMATIONS = "permissive"

    def guide(*args: Any, **kwargs: Any) -> Any:
        return dataclasses.field(default=None)

    def generable(arg: Any = None) -> Any:
        def apply(cls: type[Any]) -> type[Any]:
            if not dataclasses.is_dataclass(cls):
                cls = dataclasses.dataclass(cls)
            cls_any = cast(Any, cls)

            def generation_schema(inner_cls: type[Any]) -> GenerationSchema:
                return GenerationSchema()

            cls_any._generable = True
            cls_any.generation_schema = classmethod(generation_schema)
            return cls

        if isinstance(arg, type):
            return apply(arg)
        return apply

    mock_session_ctor = MagicMock()
    mock_session_inst = MagicMock()
    mock_session_inst.respond = AsyncMock()
    mock_session_inst.token_usage = AsyncMock(
        return_value={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    mock_session_ctor.return_value = mock_session_inst

    def fake_get_sentence_embedding(text: str) -> list[float]:
        n = 512
        return [0.01 * (len(text) % 7) + 0.001 * (i % 13) for i in range(n)]

    fake_sdk = type(
        "FakeFoundationModelsSDK",
        (),
        {
            "GenerationSchema": GenerationSchema,
            "SystemLanguageModel": SystemLanguageModel,
            "SystemLanguageModelGuardrails": SystemLanguageModelGuardrails,
            "LanguageModelSession": mock_session_ctor,
            "guide": staticmethod(guide),
            "generable": staticmethod(generable),
            "get_sentence_embedding": staticmethod(fake_get_sentence_embedding),
        },
    )()

    return {
        "sdk": fake_sdk,
        "session": mock_session_inst,
        "session_ctor": mock_session_ctor,
    }


@pytest.fixture
def client(mock_fm_sdk: dict[str, Any]) -> Generator[TestClient]:
    with TestClient(create_app(fm_sdk=mock_fm_sdk["sdk"], request_dump_path=None)) as test_client:
        yield test_client


def test_chat_completions_basic(client: TestClient, mock_fm_sdk: dict[str, Any]) -> None:
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


def test_chat_completions_streaming(client: TestClient, mock_fm_sdk: dict[str, Any]) -> None:
    # Given
    async def mock_stream(prompt: Any) -> AsyncGenerator[str]:
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


def test_chat_completions_json_schema(client: TestClient, mock_fm_sdk: dict[str, Any]) -> None:
    # Given
    from dataclasses import make_dataclass

    cat_cls = make_dataclass("Cat", [("name", str), ("age", int)])
    mock_fm_sdk["session"].respond.return_value = cat_cls(name="Whiskers", age=3)

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


def test_responses_api_basic(client: TestClient, mock_fm_sdk: dict[str, Any]) -> None:
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


def test_responses_api_streaming(client: TestClient, mock_fm_sdk: dict[str, Any]) -> None:
    # Given
    async def mock_stream(prompt: Any) -> AsyncGenerator[str]:
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


def test_responses_api_streaming_deduplicates_multiline_snapshots(
    client: TestClient,
    mock_fm_sdk: dict[str, Any],
) -> None:
    # Given
    async def mock_stream(prompt: Any) -> AsyncGenerator[str]:
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
    client: TestClient,
    mock_fm_sdk: dict[str, Any],
) -> None:
    # Given
    async def mock_stream(prompt: Any) -> AsyncGenerator[str]:
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


def test_responses_api_codex_exec_streaming_uses_final_item_content(
    client: TestClient,
    mock_fm_sdk: dict[str, Any],
) -> None:
    # Given
    async def mock_stream(prompt: Any) -> AsyncGenerator[str]:
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


def test_responses_api_json_schema(client: TestClient, mock_fm_sdk: dict[str, Any]) -> None:
    # Given
    from dataclasses import make_dataclass

    cat_cls = make_dataclass("Cat", [("name", str), ("age", int)])
    mock_fm_sdk["session"].respond.return_value = cat_cls(name="Whiskers", age=3)
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


def test_responses_api_adapts_codex_instructions(
    client: TestClient, mock_fm_sdk: dict[str, Any]
) -> None:
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


def test_build_responses_prompt_preserves_codex_conversation_history() -> None:
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


def test_responses_api_preserves_regular_instructions(
    client: TestClient, mock_fm_sdk: dict[str, Any]
) -> None:
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


def test_embeddings_openai_single_string(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        json={"input": "Hello world", "model": "text-embedding-3-small"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["model"] == "text-embedding-3-small"
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == "embedding"
    assert data["data"][0]["index"] == 0
    assert len(data["data"][0]["embedding"]) == 512
    assert data["data"][0]["embedding"][0] == 0.01 * (len("Hello world") % 7)
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"]


def test_embeddings_openai_array(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        json={"input": ["First phrase.", "Second phrase."]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2
    assert data["data"][0]["index"] == 0
    assert data["data"][1]["index"] == 1
    assert len(data["data"][0]["embedding"]) == 512
    # Different string lengths => different first scalar in fake
    assert data["data"][0]["embedding"][0] != data["data"][1]["embedding"][0]


def test_embeddings_base64(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        json={"input": "Test", "encoding_format": "base64"},
    )
    assert response.status_code == 200
    raw = base64.standard_b64decode(response.json()["data"][0]["embedding"])
    floats = struct.unpack(f"<{512}f", raw)
    assert len(floats) == 512
    assert abs(floats[0] - 0.01 * (4 % 7)) < 1e-6


def test_embeddings_errors(client: TestClient) -> None:
    assert client.post("/v1/embeddings", json={}).status_code == 400
    assert client.post("/v1/embeddings", json={"input": ""}).status_code == 400
    assert client.post("/v1/embeddings", json={"input": []}).status_code == 400
    assert client.post("/v1/embeddings", json={"input": 42}).status_code == 400
    bad_fmt = client.post(
        "/v1/embeddings",
        json={"input": "ok", "encoding_format": "float16"},
    )
    assert bad_fmt.status_code == 400


def test_build_responses_prompt_strips_environment_context_for_codex_mode() -> None:
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
