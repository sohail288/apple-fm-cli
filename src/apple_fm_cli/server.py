import json
import time
import uuid
import dataclasses
from typing import Any, AsyncGenerator

import tiktoken
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import apple_fm_sdk as fm
from apple_fm_cli import create_dynamic_dataclass

app = FastAPI(title="Apple FM OpenAI Compatibility Server")

# Use cl100k_base as a reasonable approximation for modern models
TOKENIZER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

def format_openai_chunk(
    id: str, 
    model: str, 
    content: str | None = None, 
    finish_reason: str | None = None,
    role: str | None = None
) -> str:
    chunk = {
        "id": id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }
        ]
    }
    if content is not None:
        chunk["choices"][0]["delta"]["content"] = content
    if role is not None:
        chunk["choices"][0]["delta"]["role"] = role
        
    return f"data: {json.dumps(chunk)}\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])
    model_name = body.get("model", "apple-fm")
    stream = body.get("stream", False)
    response_format = body.get("response_format")
    
    if not messages:
        raise HTTPException(status_code=400, detail="Messages list is empty")

    # Extract instructions from system/developer messages
    instructions_parts = []
    other_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "developer"):
            instructions_parts.append(content)
        else:
            other_messages.append(m)
            
    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    
    # Construct the prompt from remaining messages
    if not other_messages:
        # If only system messages were provided, we need a prompt
        raise HTTPException(status_code=400, detail="No user message provided")
        
    last_msg = other_messages[-1]
    history = other_messages[:-1]
    
    prompt_parts = []
    for m in history:
        role = m.get("role")
        content = m.get("content")
        # Format history as a simple text transcript
        role_label = "Assistant" if role == "assistant" else "User"
        prompt_parts.append(f"{role_label}: {content}")
        
    query = last_msg.get("content")
    if prompt_parts:
        full_prompt = "\n\n".join(prompt_parts) + f"\n\nUser: {query}"
    else:
        full_prompt = query

    # Handle guided generation (JSON Schema)
    generating_type = None
    if response_format and response_format.get("type") == "json_schema":
        schema_info = response_format.get("json_schema", {})
        schema = schema_info.get("schema")
        if schema:
            name = schema_info.get("name", "GeneratedObject")
            generating_type = create_dynamic_dataclass(name, schema)

    # Check for models availability
    model_checker = fm.SystemLanguageModel()
    is_available, reason = model_checker.is_available()
    if not is_available:
        raise HTTPException(status_code=503, detail=f"Foundation Models not available: {reason}")

    session = fm.LanguageModelSession(instructions=instructions)
    completion_id = f"chatcmpl-{uuid.uuid4()}"

    if stream:
        async def stream_generator() -> AsyncGenerator[str, None]:
            # Send initial role
            yield format_openai_chunk(completion_id, model_name, role="assistant")
            
            try:
                async for chunk in session.stream_response(full_prompt):
                    # chunk is typically a string or has a .text attribute
                    text = chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
                    yield format_openai_chunk(completion_id, model_name, content=text)
                
                yield format_openai_chunk(completion_id, model_name, finish_reason="stop")
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            respond_kwargs = {}
            if generating_type:
                respond_kwargs["generating"] = generating_type
                
            response = await session.respond(full_prompt, **respond_kwargs)
            
            # Format text content
            if generating_type:
                # response is a dataclass instance
                content = json.dumps(dataclasses.asdict(response))
            else:
                # response has a .text attribute
                content = getattr(response, "text", str(response))
            
            prompt_tokens = count_tokens(full_prompt) + count_tokens(instructions or "")
            completion_tokens = count_tokens(content)
                
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)
