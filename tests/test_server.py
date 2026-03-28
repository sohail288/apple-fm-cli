import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch

from apple_fm_cli.server import app

client = TestClient(app)

@pytest.fixture
def mock_fm_sdk():
    with patch("apple_fm_sdk.SystemLanguageModel") as mock_model, \
         patch("apple_fm_sdk.LanguageModelSession") as mock_session:
        
        mock_model_inst = MagicMock()
        mock_model_inst.is_available.return_value = (True, "")
        mock_model.return_value = mock_model_inst
        
        mock_session_inst = MagicMock()
        mock_session_inst.respond = AsyncMock()
        mock_session.return_value = mock_session_inst
        
        yield {
            "model": mock_model_inst,
            "session": mock_session_inst
        }

def test_chat_completions_basic(mock_fm_sdk):
    mock_fm_sdk["session"].respond.return_value = MagicMock(text="Hello world")
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "apple-fm"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello world"
    assert data["object"] == "chat.completion"
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0

def test_chat_completions_streaming(mock_fm_sdk):
    # Mock stream_response
    async def mock_stream(prompt):
        yield "Hello "
        yield "world"
    
    mock_fm_sdk["session"].stream_response = mock_stream
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "apple-fm",
            "stream": True
        }
    )
    
    assert response.status_code == 200
    # Collect SSE chunks
    lines = [line for line in response.text.split("\n") if line.strip()]
    assert len(lines) > 3
    assert "Hello " in lines[1]
    assert "world" in lines[2]
    assert "[DONE]" in lines[-1]

def test_chat_completions_json_schema(mock_fm_sdk):
    from dataclasses import make_dataclass
    Cat = make_dataclass("Cat", [("name", str), ("age", int)])
    mock_fm_sdk["session"].respond.return_value = Cat(name="Whiskers", age=3)
    
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
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        }
                    }
                }
            }
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    content = json.loads(data["choices"][0]["message"]["content"])
    assert content["name"] == "Whiskers"
    assert content["age"] == 3
