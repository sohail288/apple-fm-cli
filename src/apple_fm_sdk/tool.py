# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import asyncio
import threading
import logging
from .generation_schema import GenerationSchema
from .generable import GeneratedContent
from .c_helpers import _ManagedObject, _get_error_string
import ctypes
from abc import ABC, abstractmethod
from .errors import _status_code_to_exception

logger = logging.getLogger(__name__)

try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise ImportError(
        "Foundation Models C bindings not found. Please ensure _foundationmodels_ctypes.py is available."
    )


class Tool(_ManagedObject, ABC):
    """Base class for creating tools that foundation models can invoke during generation.

    A ``Tool`` bridges Python async functions to the Foundation Models API, enabling
    foundation models to perform actions like calculations, API calls, database queries,
    or any other programmatic operations during generation.

    **Tool Lifecycle:**

    1. **Definition**: Subclass Tool and implement required methods/properties
    2. **Registration**: Pass tool instances to LanguageModelSession
    3. **Invocation**: Model automatically calls tools when appropriate
    4. **Execution**: Your async ``call()`` method executes with parsed arguments
    5. **Response**: Tool result is returned to the model to continue generation

    **Callback Mechanism:**

    Tools use an async callback system that:

    - Automatically handles argument parsing from GeneratedContent
    - Executes your ``call()`` method in the appropriate async context
    - Manages threading and event loops transparently
    - Returns results or errors back to the model

    **Async Requirements:**

    The ``call()`` method MUST be an async function (coroutine). This allows tools to:

    - Make async API calls without blocking
    - Perform I/O operations efficiently
    - Run concurrent operations when needed
    - Integrate with async frameworks

    **Error Handling:**

    - Exceptions in ``call()`` are caught and reported to the model
    - The model receives error messages and can adapt its response
    - Tools should raise descriptive exceptions for better model understanding

    Examples:
        Simple calculator tool::

            import apple_fm_sdk as fm

            @fm.generable("Calculator parameters")
            class CalculatorParams:
                operation: str = fm.guide("The operation to perform")
                a: float = fm.guide("First number")
                b: float = fm.guide("Second number")

            class CalculatorTool(fm.Tool):
                name = "calculator"
                description = "Performs basic arithmetic operations"

                @property
                def arguments_schema(self) -> fm.GenerationSchema:
                    return CalculatorParams.generation_schema()

                async def call(self, args: fm.GeneratedContent) -> str:
                    op = args.value(str, for_property="operation")
                    a = args.value(float, for_property="a")
                    b = args.value(float, for_property="b")

                    if op == "add":
                        result = a + b
                    elif op == "multiply":
                        result = a * b
                    else:
                        raise ValueError(f"Unknown operation: {op}")

                    return str(result)

        Tool with async API call::

            import aiohttp
            import apple_fm_sdk as fm

            @fm.generable("Weather parameters")
            class WeatherParams:
                city: str = fm.guide("The city to get weather for")
                units: str = fm.guide("Temperature units (metric or imperial)")

            class WeatherTool(fm.Tool):
                name = "get_weather"
                description = "Gets current weather for a city"

                @property
                def arguments_schema(self) -> fm.GenerationSchema:
                    return WeatherParams.generation_schema()

                async def call(self, args: fm.GeneratedContent) -> str:
                    city = args.value(str, for_property="city")
                    try:
                        units = args.value(str, for_property="units")
                    except Exception:
                        units = "metric"

                    # Implement async API call to fetch weather here
                    return "Sunny, 25°C"  # Placeholder response

        Tool with error handling::

            import apple_fm_sdk as fm

            @fm.generable("Database query parameters")
            class DatabaseParams:
                user_id: int = fm.guide("The user ID to query")

            class DatabaseTool(fm.Tool):
                name = "query_database"
                description = "Queries the user database"

                @property
                def arguments_schema(self) -> fm.GenerationSchema:
                    return DatabaseParams.generation_schema()

                async def call(self, args: fm.GeneratedContent) -> str:
                    user_id = args.value(int, for_property="user_id")
                    # Implement database query with error handling here
                    return f"User data for ID {user_id}"  # Placeholder response

        Using tools in a session::

            from apple_fm_sdk import LanguageModelSession

            session = LanguageModelSession(
                instructions="You are a helpful assistant with access to tools.",
                tools=[CalculatorTool(), WeatherTool(), DatabaseTool()]
            )

            # Model will automatically use tools when appropriate
            response = await session.respond("What's 15% of 240?")
            # Model invokes CalculatorTool internally

    Attributes:
        name: The tool's name (must be set by subclass)
        description: Human-readable description of what the tool does (must be set by subclass)

    Note:
        - Tool names should be descriptive and follow snake_case convention
        - Descriptions should explain the tool's purpose and when to use it
        - The ``call()`` method must be async even if it doesn't perform async operations
        - Tools are automatically managed by the session's lifecycle
        - Multiple tools can be registered with a single session

    See Also:
        - :class:`~apple_fm_sdk.session.LanguageModelSession`: For using tools in sessions
        - :class:`~apple_fm_sdk.generation_schema.GenerationSchema`: For defining argument schemas
        - :class:`~apple_fm_sdk.generable.GeneratedContent`: For accessing parsed arguments
    """

    name: str
    description: str

    @property
    @abstractmethod
    def arguments_schema(self) -> GenerationSchema:
        """Define the schema for tool arguments.

        This property must return a GenerationSchema that describes the structure
        and types of arguments the tool expects. The model uses this schema to
        generate properly formatted arguments when invoking the tool.

        :return: Schema defining the tool's expected arguments
        :rtype: GenerationSchema

        Example:
            ::

                import apple_fm_sdk as fm

                @fm.generable("Search parameters")
                class SearchParams:
                    query: str = fm.guide("The search query")
                    limit: int = fm.guide("Maximum number of results")

                @property
                def arguments_schema(self) -> fm.GenerationSchema:
                    return SearchParams.generation_schema()
        """
        pass

    @abstractmethod
    async def call(self, args: GeneratedContent) -> str:
        """Execute the tool's functionality with the provided arguments.

        This async method is invoked when the model decides to use the tool.
        The arguments are automatically parsed according to the ``arguments_schema``
        and provided as a GeneratedContent object.

        :param args: Parsed arguments as GeneratedContent. Access values via ``args.value``
            which contains a dictionary matching your schema structure.
        :type args: GeneratedContent
        :return: The tool's result as a string. This result is provided back to the
            model to inform its continued generation.
        :rtype: str
        :raises Exception: Any exception raised will be caught and reported to the model
            as an error message. Use descriptive exceptions to help the model
            understand what went wrong.

        Example:
            ::

                async def call(self, args: fm.GeneratedContent) -> str:
                    query = args.value(str, for_property="query")
                    try:
                        limit = args.value(int, for_property="limit")
                    except Exception:
                        limit = 10

                    # Perform async operation, for example, database search or another session call here

                    return f"Results for '{query}' with limit {limit}"  # Placeholder response

        Note:
            - Must be an async function even if no async operations are performed
            - Return value must be a string (convert other types as needed)
            - Exceptions are automatically handled and reported to the model
        """
        pass

    def __init__(self):
        # Verify the subclass implementation
        self._verify_subclass_()

        # Store the async callable
        self._async_callable = self.call
        self._pending_calls = {}  # Maps call_id to future
        self._call_lock = threading.Lock()

        # Create the C callback function type matching the bindings
        # UNCHECKED(None) in the bindings returns ctypes.c_void_p
        CallbackType = ctypes.CFUNCTYPE(
            ctypes.c_void_p, lib.FMGeneratedContentRef, ctypes.c_uint
        )

        # Create the actual callback function
        def _c_callback_impl(content_ref, call_id):
            """C callback that gets invoked when the tool is called."""
            try:
                # Create GeneratedContent from the C pointer
                # Swift passes the pointer with ownership already transferred (passRetained)
                # so we don't need to manually retain it here
                generated_content = GeneratedContent(_ptr=content_ref)

                # Run the async callable in a new task
                async def _run_async_callable():
                    try:
                        # Call the tool subclass's async function
                        result = await self._async_callable(generated_content)

                        # Convert result to string if needed
                        if not isinstance(result, str):
                            result = str(result)

                        # Finish the tool call with the result
                        result_bytes = result.encode("utf-8")
                        lib.FMBridgedToolFinishCall(self._ptr, call_id, result_bytes)

                    except Exception as e:
                        # On error, finish with error message
                        error_msg = f"Tool error: {str(e)}"
                        error_bytes = error_msg.encode("utf-8")
                        lib.FMBridgedToolFinishCall(self._ptr, call_id, error_bytes)

                # Schedule the async callable
                # Try to get the current running loop, or create a new one
                try:
                    loop = asyncio.get_running_loop()  # noqa: F841 this unused variable is needed to check if a loop is running
                    asyncio.create_task(_run_async_callable())
                except RuntimeError:
                    # No running loop - create a new thread with event loop
                    def _run_in_thread():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(_run_async_callable())
                        finally:
                            loop.close()

                    thread = threading.Thread(target=_run_in_thread, daemon=True)
                    thread.start()

            except Exception as e:
                # Catch-all error handler
                error_msg = f"Callback error: {str(e)}"
                error_bytes = error_msg.encode("utf-8")
                try:
                    lib.FMBridgedToolFinishCall(self._ptr, call_id, error_bytes)
                except Exception:
                    raise

        # Wrap the callback implementation with the callback type
        _c_callback = CallbackType(_c_callback_impl)

        # Store the callback to prevent garbage collection
        self._c_callback = _c_callback

        # Initialize _ptr to None before calling super().__init__() to avoid AttributeError in __del__
        self._ptr = None

        # Create the bridged tool using the C API
        name_bytes = self.name.encode("utf-8")
        description_bytes = self.description.encode("utf-8")

        # Store the schema to keep it alive (prevents deallocation before FMBridgedToolCreate completes)
        # This is necessary because arguments_schema is a property that returns a new object each time
        self._arguments_schema = self.arguments_schema

        # Prepare error handling parameters
        error_code = ctypes.c_int()
        error_description = ctypes.POINTER(ctypes.c_char)()

        ptr = lib.FMBridgedToolCreate(
            name_bytes,
            description_bytes,
            self._arguments_schema._ptr,
            self._c_callback,
            ctypes.byref(error_code),
            ctypes.byref(error_description),
        )

        # Check for errors
        if not ptr:
            err_code, err_desc = _get_error_string(error_code, error_description)
            error_msg = "Failed to create bridged tool"
            if err_desc:
                error_msg = error_msg + ": " + err_desc
            raise _status_code_to_exception(err_code or error_code.value, error_msg)

        super().__init__(ptr)

    def _verify_subclass_(self):
        assert hasattr(self, "name"), "Tool subclass must have a 'name' property."
        assert hasattr(self, "description"), (
            "Tool subclass must have a 'description' property."
        )
        assert hasattr(self, "arguments_schema"), (
            "Tool subclass must have an 'arguments_schema' property."
        )
        assert hasattr(self, "call"), "Tool subclass must implement the 'call' method."
        if not isinstance(self.name, str):
            raise TypeError("Tool name must be a string.")
        if not isinstance(self.description, str):
            raise TypeError("Tool description must be a string.")
        if not isinstance(self.arguments_schema, GenerationSchema):
            raise TypeError(
                "Tool arguments_schema must be a GenerationSchema instance."
            )
        if not asyncio.iscoroutinefunction(self.call):
            raise TypeError("Tool call method must be an async function.")
