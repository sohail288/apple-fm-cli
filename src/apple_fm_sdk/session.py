# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import asyncio
import ctypes
import json
import logging
import queue
import threading
import warnings
from typing import Any, AsyncIterator, Optional, Type, Union, overload

from apple_fm_sdk.transcript import Transcript

from .c_helpers import (
    StreamingCallback,
    _ManagedObject,
    _register_handle,
    _session_callback,
    _session_structured_callback,
    _unregister_handle,
)
from .content import AudioContentPart, ContentPart, ImageContentPart
from .core import SystemLanguageModel, SystemLanguageModelUseCase
from .errors import FoundationModelsError
from .generable import Generable, GeneratedContent
from .generation_options import GenerationOptions
from .generation_schema import GenerationSchema
from .tool import Tool

logger = logging.getLogger(__name__)

try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise ImportError(
        "Foundation Models C bindings not found. Please ensure _foundationmodels_ctypes.py is available."
    )

Prompt = str  # Alias for prompt type


class LanguageModelSession(_ManagedObject):
    """Represents a language model session for foundation model interactions."""

    def __init__(
        self,
        instructions: Optional[str] = None,
        model: Optional[SystemLanguageModel] = None,
        tools: Optional[list[Tool]] = None,
        _ptr=None,
    ):
        """Create a language model session."""
        # Initialize request lock for preventing concurrent requests
        self._request_lock = asyncio.Lock()
        self._active_task = None
        self._model = model if model else SystemLanguageModel()

        if _ptr is not None:
            # Internal constructor for specific ptr
            super().__init__(_ptr)
        else:
            # Create model pointer
            model_ptr = self._model._ptr

            # Encode instructions if provided
            instructions_cstr = None
            if instructions:
                instructions_cstr = instructions.encode("utf-8")

            # Create array of tool pointers
            tool_count = len(tools) if tools else 0
            tool_refs = (ctypes.c_void_p * tool_count)()
            if tools:
                for i, tool in enumerate(tools):
                    tool_refs[i] = tool._ptr

            # Create the session via C binding
            ptr = lib.FMLanguageModelSessionCreateFromSystemLanguageModel(
                model_ptr, instructions_cstr, tool_refs, tool_count
            )

            # Create transcript
            self.transcript = Transcript(ptr)

            super().__init__(ptr)

    @staticmethod
    def summarization(instructions: Optional[str] = None) -> "LanguageModelSession":
        """Create a session optimized for summarization tasks."""
        model = SystemLanguageModel(use_case=SystemLanguageModelUseCase.GENERAL)
        return LanguageModelSession(instructions=instructions, model=model)

    @staticmethod
    def content_tagging(instructions: Optional[str] = None) -> "LanguageModelSession":
        """Create a session optimized for content tagging and classification."""
        model = SystemLanguageModel(use_case=SystemLanguageModelUseCase.CONTENT_TAGGING)
        return LanguageModelSession(instructions=instructions, model=model)

    @staticmethod
    def proofreading(instructions: Optional[str] = None) -> "LanguageModelSession":
        """Create a session optimized for proofreading and grammatical correction."""
        return LanguageModelSession(instructions=instructions)

    def token_count(self, text: str) -> int:
        """Measure how many tokens a string uses in the current session."""
        return self._model.token_count(text)

    async def token_usage(self) -> dict[str, int]:
        """Get the current token usage for the entire session transcript."""
        t_dict = await self.transcript.to_dict()
        entries = t_dict.get("transcript", {}).get("entries", [])

        usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "instructions_tokens": 0,
        }

        for entry in entries:
            role = entry.get("role")
            text_content = ""
            for content in entry.get("contents", []):
                if content.get("type") == "text":
                    text_content += content.get("text", "")

            count = self.token_count(text_content)

            if role == "instructions":
                usage["instructions_tokens"] += count
            elif role == "user":
                usage["prompt_tokens"] += count
            elif role == "response":
                usage["completion_tokens"] += count

        usage["total_tokens"] = sum(v for k, v in usage.items() if k != "total_tokens")
        return usage

    @classmethod
    def from_transcript(
        cls,
        transcript: Transcript,
        model: Optional[SystemLanguageModel] = None,
        tools: Optional[list[Tool]] = None,
    ) -> "LanguageModelSession":
        """Create a new session from an existing transcript."""
        # Create model pointer
        model_ptr = model._ptr if model else None

        # Create array of tool pointers
        tool_count = len(tools) if tools else 0
        tool_refs = (ctypes.c_void_p * tool_count)()
        if tools:
            for i, tool in enumerate(tools):
                tool_refs[i] = tool._ptr

        # Create the session via C binding
        ptr = lib.FMLanguageModelSessionCreateFromTranscript(
            transcript.session_ptr, model_ptr, tool_refs, tool_count
        )

        # Update transcript to use the new session pointer
        transcript._update_session_ptr(ptr)

        # Create session instance
        session = cls(_ptr=ptr)
        session.transcript = transcript
        return session

    @property
    def is_responding(self) -> bool:
        """Check if the session is currently responding."""
        return lib.FMLanguageModelSessionIsResponding(self._ptr)

    def _reset_task_state(self):
        """Reset internal task handling state."""
        lib.FMLanguageModelSessionReset(self._ptr)

    @overload
    async def respond(
        self, prompt: Union[str, list[ContentPart]], *, options: Optional[GenerationOptions] = None
    ) -> str: ...

    @overload
    async def respond(
        self,
        prompt: Union[str, list[ContentPart]],
        *,
        generating: type[Generable],
        options: Optional[GenerationOptions] = None,
    ) -> Type[Any]: ...

    async def respond(
        self,
        prompt: Union[str, list[ContentPart]],
        generating: Optional[Union[Type[Generable], Generable]] = None,
        *,
        schema: Optional[GenerationSchema] = None,
        json_schema: Optional[dict] = None,
        options: Optional[GenerationOptions] = None,
    ) -> Union[str, Any, GeneratedContent]:
        """Get a response to a prompt with optional guided generation."""

        # Handle multimodal prompt
        final_prompt = ""
        if isinstance(prompt, list):
            for part in prompt:
                if isinstance(part, str):
                    final_prompt += part
                elif isinstance(part, (ImageContentPart, AudioContentPart)):
                    warnings.warn(
                        f"Multimodal part {type(part).__name__} is currently not supported "
                        "by the native bridge and will be treated as a placeholder.",
                        UserWarning,
                        stacklevel=2,
                    )
                    final_prompt += f"\n[{type(part).__name__} Input]\n"
            prompt = final_prompt

        # Validate arguments
        if generating is not None and schema is not None:
            raise ValueError("Cannot specify both 'generating' and 'schema' arguments")

        # Handle guided generation with generable type
        if generating is not None:
            if not isinstance(generating, Generable):
                raise ValueError(
                    f"{generating.__name__} is not a Generable type. Use @generable decorator."
                )
            gen_schema = generating.generation_schema()
            generated_content = await self._respond_with_schema(prompt, gen_schema, options)
            return generating._from_generated_content(generated_content)

        # Handle guided generation with explicit schema
        if schema is not None:
            return await self._respond_with_schema(prompt, schema, options)

        # Handle guided generation from raw JSON schema string
        if json_schema is not None:
            return await self._respond_with_schema_from_json(prompt, json_schema, options)

        # Handle basic text response
        return await self._respond_basic(prompt, options)

    async def _respond_basic(self, prompt: str, options: Optional[GenerationOptions] = None) -> str:
        """Get a complete basic text response to a prompt."""
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            prompt_bytes = prompt.encode("utf-8")
            options_json = json.dumps(options.to_dict()).encode("utf-8") if options else None
            future_handle = _register_handle(future)
            task = lib.FMLanguageModelSessionRespond(
                self._ptr, prompt_bytes, options_json, future_handle, _session_callback
            )
            self._active_task = task
            try:
                await future
            except asyncio.CancelledError as e:
                lib.FMTaskCancel(task)
                future.cancel()
                max_wait_time = 1.0
                poll_interval = 0.01
                elapsed = 0.0
                while self.is_responding and elapsed < max_wait_time:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                self._reset_task_state()
                raise e
            finally:
                _unregister_handle(future_handle)
                lib.FMRelease(task)
                self._active_task = None
            return future.result()

    async def _respond_with_schema(
        self,
        prompt: str,
        schema: GenerationSchema,
        options: Optional[GenerationOptions] = None,
    ) -> GeneratedContent:
        """Internal method for guided generation using a GenerationSchema."""
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            prompt_bytes = prompt.encode("utf-8")
            options_json = json.dumps(options.to_dict()).encode("utf-8") if options else None
            future_handle = _register_handle(future)
            task = lib.FMLanguageModelSessionRespondWithSchema(
                self._ptr,
                prompt_bytes,
                schema._ptr,
                options_json,
                future_handle,
                _session_structured_callback,
            )
            self._active_task = task
            try:
                await future
            except asyncio.CancelledError as e:
                lib.FMTaskCancel(task)
                future.cancel()
                max_wait_time = 1.0
                poll_interval = 0.01
                elapsed = 0.0
                while self.is_responding and elapsed < max_wait_time:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                self._reset_task_state()
                raise e
            except Exception as e:
                self._reset_task_state()
                raise e
            finally:
                _unregister_handle(future_handle)
                lib.FMRelease(task)
                self._active_task = None
            return future.result()

    async def _respond_with_schema_from_json(
        self,
        prompt: str,
        json_schema: dict,
        options: Optional[GenerationOptions] = None,
    ) -> GeneratedContent:
        """Internal method for guided generation using a JSON schema string."""
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            prompt_bytes = prompt.encode("utf-8")
            json_schema_bytes = json.dumps(json_schema).encode("utf-8")
            options_json = json.dumps(options.to_dict()).encode("utf-8") if options else None
            future_handle = _register_handle(future)
            task = lib.FMLanguageModelSessionRespondWithSchemaFromJSON(
                self._ptr,
                prompt_bytes,
                json_schema_bytes,
                options_json,
                future_handle,
                _session_structured_callback,
            )
            self._active_task = task
            try:
                await future
            except asyncio.CancelledError as e:
                lib.FMTaskCancel(task)
                future.cancel()
                max_wait_time = 1.0
                poll_interval = 0.01
                elapsed = 0.0
                while self.is_responding and elapsed < max_wait_time:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                self._reset_task_state()
                raise e
            except Exception as e:
                self._reset_task_state()
                raise e
            finally:
                _unregister_handle(future_handle)
                lib.FMRelease(task)
                self._active_task = None
            return future.result()

    async def stream_response(
        self, prompt: Union[str, list[ContentPart]], options: Optional[GenerationOptions] = None
    ) -> AsyncIterator[str]:
        """Stream response chunks for a prompt (text only)."""
        # Handle multimodal prompt
        final_prompt = ""
        if isinstance(prompt, list):
            for part in prompt:
                if isinstance(part, str):
                    final_prompt += part
                elif isinstance(part, (ImageContentPart, AudioContentPart)):
                    warnings.warn(
                        f"Multimodal part {type(part).__name__} is currently not supported "
                        "by streaming and will be treated as a placeholder.",
                        UserWarning,
                        stacklevel=2,
                    )
                    final_prompt += f"\n[{type(part).__name__} Input]\n"
            prompt = final_prompt

        async for chunk in self._stream_response_basic(prompt, options):
            yield chunk

    async def _stream_response_basic(
        self, prompt: Prompt, options: Optional[GenerationOptions] = None
    ) -> AsyncIterator[str]:
        """Stream basic text response chunks for a prompt."""
        callback = StreamingCallback()
        stream_thread = None
        stream_ptr_holder = [None]

        def _start_stream():
            prompt_bytes = prompt.encode("utf-8")
            options_json = json.dumps(options.to_dict()).encode("utf-8") if options else None
            stream_ptr = lib.FMLanguageModelSessionStreamResponse(
                self._ptr, prompt_bytes, options_json
            )
            stream_ptr_holder[0] = stream_ptr
            if not stream_ptr:
                callback.error = FoundationModelsError("Failed to create response stream")
                callback.queue.put(None)
                callback.completed.set()
                return
            try:
                lib.FMLanguageModelSessionResponseStreamIterate(
                    stream_ptr, None, callback._callback
                )
            except Exception as e:
                callback.error = FoundationModelsError(f"Stream iteration error: {e}")
                callback.queue.put(None)
                callback.completed.set()

        try:
            stream_thread = threading.Thread(target=_start_stream)
            stream_thread.daemon = True
            stream_thread.start()
            while True:
                try:
                    snapshot = callback.queue.get(timeout=0.1)
                    if snapshot is None:
                        break
                    yield snapshot
                except queue.Empty:
                    if callback.completed.is_set():
                        try:
                            while True:
                                snapshot = callback.queue.get_nowait()
                                if snapshot is None:
                                    break
                                yield snapshot
                        except queue.Empty:
                            pass
                        break
                    continue
            if callback.error:
                raise callback.error
        finally:
            if stream_thread and stream_thread.is_alive():
                stream_thread.join(timeout=2.0)
            if stream_ptr_holder[0]:
                lib.FMRelease(stream_ptr_holder[0])
