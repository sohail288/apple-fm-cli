import dataclasses
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

import apple_fm_sdk as fm
from apple_fm_cli import create_dynamic_dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Apple FM OpenAI Compatibility Server")

CODEX_PROMPT_MARKERS = (
    "You are a coding agent running in the Codex CLI",
    "Codex CLI is an open source project led by OpenAI",
)

APPLE_FM_CODEX_INSTRUCTIONS = (
    "You are a concise assistant.\n"
    "Answer only the user's actual request.\n"
    "Do not describe plans, repo exploration, tools, patches, or tests unless explicitly asked.\n"
    "Ignore environment metadata unless the user asks about it.\n"
    "For simple factual or math questions, respond with just the answer.\n"
    "If the request is ambiguous, ask one short clarifying question instead of guessing.\n"
    "Keep the response brief and literal."
)


def format_sse_event(event: str, payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload)
    logger.info(f"Outgoing SSE event ({event}): {payload_json}")
    return f"event: {event}\ndata: {payload_json}\n\n"


def format_openai_chunk(
    id: str,
    model: str,
    content: str | None = None,
    finish_reason: str | None = None,
    role: str | None = None,
    object_type: str = "chat.completion.chunk",
    message_id: str | None = None,
    usage: dict | None = None,
    event: str | None = None,
    status: str | None = None,
) -> str:
    if object_type == "chat.completion.chunk":
        chunk = {
            "id": id,
            "object": object_type,
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        if content is not None:
            chunk["choices"][0]["delta"]["content"] = content
        if role is not None:
            chunk["choices"][0]["delta"]["role"] = role
        if usage is not None:
            chunk["usage"] = usage
    else:
        # response format
        m_id = message_id or f"msg_{uuid.uuid4()}"
        chunk = {
            "id": id,
            "object": "response",
            "created_at": int(time.time()),
            "model": model,
            "status": status or ("completed" if finish_reason == "stop" else "in_progress"),
            "output": [{"id": m_id, "type": "message", "delta": {}}],
        }
        if content is not None:
            chunk["output"][0]["delta"]["content"] = [{"type": "output_text", "text": content}]
        if role is not None:
            chunk["output"][0]["role"] = role
        if finish_reason is not None:
            chunk["output"][0]["status"] = "completed" if finish_reason == "stop" else finish_reason
        if usage is not None:
            chunk["usage"] = usage

    chunk_json = json.dumps(chunk)
    logger.info(f"Outgoing chunk (event: {event}): {chunk_json}")

    out = ""
    if event:
        out += f"event: {event}\n"
    out += f"data: {chunk_json}\n\n"
    return out


def truncate_text(text: str, max_tokens: int = 3000) -> str:
    # Heuristic: 1 token ~= 4 chars
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated] ..."
    return text


def format_responses_usage(usage_data: dict[str, int]) -> dict[str, int]:
    input_tokens = usage_data.get("prompt_tokens", 0) + usage_data.get("instructions_tokens", 0)
    output_tokens = usage_data.get("completion_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": 0,
        "total_tokens": usage_data.get("total_tokens", input_tokens + output_tokens),
    }


def incremental_text(snapshot_text: str, previous_text: str) -> tuple[str, str]:
    if snapshot_text.startswith(previous_text):
        delta = snapshot_text[len(previous_text) :]
        return delta, snapshot_text
    return snapshot_text, snapshot_text


def adapt_codex_instructions(instructions: str | None) -> str | None:
    if not instructions:
        return instructions
    if any(marker in instructions for marker in CODEX_PROMPT_MARKERS):
        logger.info("Adapting Codex CLI instructions for Apple FM provider")
        return APPLE_FM_CODEX_INSTRUCTIONS
    return instructions


def is_codex_instructions(instructions: str | None) -> bool:
    return bool(instructions and any(marker in instructions for marker in CODEX_PROMPT_MARKERS))


def extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def strip_environment_context(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("<environment_context>") and stripped.endswith("</environment_context>"):
        return ""
    return stripped


def build_responses_prompt(input_data: Any, *, codex_mode: bool) -> str:
    if isinstance(input_data, str):
        return input_data
    if not isinstance(input_data, list):
        raise HTTPException(status_code=400, detail="Input must be a string or array of items")

    if codex_mode:
        user_texts: list[str] = []
        for item in input_data:
            if item.get("role") != "user":
                continue
            text = strip_environment_context(extract_content_text(item.get("content")))
            if text:
                user_texts.append(text)
        return user_texts[-1] if user_texts else ""

    prompt_parts = []
    for item in input_data:
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        content = extract_content_text(item.get("content"))
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        prompt_parts.append(f"{label}: {content}")
    return "\n\n".join(prompt_parts)


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
    if instructions:
        instructions = truncate_text(instructions, 3500)
    instructions = adapt_codex_instructions(instructions)

    if not other_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    last_msg = other_messages[-1]
    history = other_messages[:-1]

    prompt_parts = []
    for m in history:
        role = m.get("role")
        content = m.get("content")
        role_label = "Assistant" if role == "assistant" else "User"
        prompt_parts.append(f"{role_label}: {content}")

    query = last_msg.get("content")
    if prompt_parts:
        full_prompt = "\n\n".join(prompt_parts) + f"\n\nUser: {query}"
    else:
        full_prompt = query

    full_prompt = truncate_text(full_prompt, 500)

    generating_type = None
    if response_format and response_format.get("type") == "json_schema":
        schema_info = response_format.get("json_schema", {})
        schema = schema_info.get("schema")
        if schema:
            name = schema_info.get("name", "GeneratedObject")
            generating_type = create_dynamic_dataclass(name, schema)

    model_checker = fm.SystemLanguageModel()
    is_available, reason = model_checker.is_available()
    if not is_available:
        raise HTTPException(status_code=503, detail=f"Foundation Models not available: {reason}")

    session = fm.LanguageModelSession(instructions=instructions)
    completion_id = f"chatcmpl-{uuid.uuid4()}"

    if stream:

        async def stream_generator() -> AsyncGenerator[str, None]:
            yield format_openai_chunk(completion_id, model_name, role="assistant")
            try:
                previous_text = ""
                async for chunk in session.stream_response(full_prompt):
                    text = chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
                    delta, previous_text = incremental_text(text, previous_text)
                    if delta:
                        yield format_openai_chunk(completion_id, model_name, content=delta)

                usage_data = await session.token_usage()
                usage = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0)
                    + usage_data.get("instructions_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                }
                yield format_openai_chunk(
                    completion_id, model_name, finish_reason="stop", usage=usage
                )
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in stream: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            respond_kwargs = {}
            if generating_type:
                respond_kwargs["generating"] = generating_type
            response = await session.respond(full_prompt, **respond_kwargs)
            if generating_type:
                content = json.dumps(dataclasses.asdict(response))
            else:
                content = getattr(response, "text", str(response))
            usage_data = await session.token_usage()
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0)
                    + usage_data.get("instructions_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/responses")
async def responses_api(request: Request) -> Any:
    raw_body = await request.body()
    with open("last_request.json", "wb") as f:
        f.write(raw_body)
    logger.info(f"Incoming Responses API raw body length: {len(raw_body)}")
    try:
        body = json.loads(raw_body)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body")

    model_name = body.get("model", "apple-fm")
    input_data = body.get("input")
    instructions = body.get("instructions")
    stream = body.get("stream", False)
    text_config = body.get("text", {})
    response_format = text_config.get("format")
    codex_mode = is_codex_instructions(instructions)

    if input_data is None:
        raise HTTPException(status_code=400, detail="Input is required")

    if instructions:
        instructions = truncate_text(instructions, 3500)
    instructions = adapt_codex_instructions(instructions)
    full_prompt = build_responses_prompt(input_data, codex_mode=codex_mode)

    full_prompt = truncate_text(full_prompt, 500)

    generating_type = None
    if response_format and response_format.get("type") == "json_schema":
        schema_info = response_format.get("json_schema", {})
        schema = schema_info.get("schema")
        if schema:
            name = schema_info.get("name", "GeneratedObject")
            generating_type = create_dynamic_dataclass(name, schema)

    model_checker = fm.SystemLanguageModel()
    is_available, reason = model_checker.is_available()
    if not is_available:
        raise HTTPException(status_code=503, detail=f"Foundation Models not available: {reason}")

    session_model = None
    if codex_mode:
        session_model = fm.SystemLanguageModel(
            guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
        )
    session = fm.LanguageModelSession(instructions=instructions, model=session_model)
    response_id = f"resp_{uuid.uuid4()}"
    message_id = f"msg_{uuid.uuid4()}"

    if stream:

        async def stream_generator() -> AsyncGenerator[str, None]:
            logger.info("Starting Responses stream")
            response_stub = {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": model_name,
                "status": "in_progress",
                "output": [],
            }
            yield format_sse_event(
                "response.created",
                {"type": "response.created", "response": response_stub},
            )
            yield format_sse_event(
                "response.in_progress",
                {"type": "response.in_progress", "response": response_stub},
            )
            yield format_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": 0,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
            )
            collected_text = ""
            try:
                async for chunk in session.stream_response(full_prompt):
                    text = chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
                    delta, collected_text = incremental_text(text, collected_text)
                    if not delta:
                        continue
                    yield format_sse_event(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "output_index": 0,
                            "item_id": message_id,
                            "content_index": 0,
                            "delta": delta,
                        },
                    )

                usage_data = await session.token_usage()
                usage = format_responses_usage(usage_data)
                completed_item = {
                    "id": message_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": (
                        []
                        if codex_mode
                        else [
                            {
                                "type": "output_text",
                                "text": collected_text,
                            }
                        ]
                    ),
                }
                completed_response = {
                    "id": response_id,
                    "object": "response",
                    "created_at": response_stub["created_at"],
                    "model": model_name,
                    "status": "completed",
                    "output": [],
                    "usage": usage,
                }
                yield format_sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": 0,
                        "item": completed_item,
                    },
                )
                yield format_sse_event(
                    "response.completed",
                    {"type": "response.completed", "response": completed_response},
                )
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in Responses stream: {e}")
                failed_response = {
                    "id": response_id,
                    "object": "response",
                    "created_at": response_stub["created_at"],
                    "model": model_name,
                    "status": "failed",
                    "output": [],
                    "error": {
                        "message": str(e),
                    },
                }
                yield format_sse_event(
                    "response.completed",
                    {"type": "response.completed", "response": failed_response},
                )
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            respond_kwargs = {}
            if generating_type:
                respond_kwargs["generating"] = generating_type
            response = await session.respond(full_prompt, **respond_kwargs)
            if generating_type:
                content = json.dumps(dataclasses.asdict(response))
            else:
                content = getattr(response, "text", str(response))
            usage_data = await session.token_usage()
            return {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": model_name,
                "status": "completed",
                "output": [
                    {
                        "id": message_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                ],
                "usage": {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0)
                    + usage_data.get("instructions_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)
