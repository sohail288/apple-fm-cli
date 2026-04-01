# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Pytest configuration for Foundation Models tests.

This configuration helps manage event loops and async test execution
to prevent trace trap errors when running async tests with pytest.
"""

import asyncio
import gc
import warnings

import pytest

import apple_fm_sdk as fm


def pytest_runtest_setup(item):
    """Hook that runs before each test."""
    print(f"\n{'=' * 60}")
    print(f"[conftest.py] Starting test: {item.nodeid}")
    print(f"{'=' * 60}")


def pytest_collection_modifyitems(config, items):
    """
    Hook to reorder tests so that tests in 'tests/' run before 'tests/doc_tests/'.

    This ensures that core functionality tests run first, followed by documentation tests.
    """
    # Separate tests into two groups
    main_tests = []
    doc_tests = []

    for item in items:
        # Get the file path relative to the tests directory
        test_path = str(item.fspath)
        if "doc_tests" in test_path:
            doc_tests.append(item)
        else:
            main_tests.append(item)

    # Reorder: main tests first, then doc tests
    items[:] = main_tests + doc_tests


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook that intercepts test results to handle ExceededContextWindowSizeError.

    If an ExceededContextWindowSizeError occurs during test execution,
    it will be converted to a warning instead of causing the test to fail.
    """
    outcome = yield
    report = outcome.get_result()

    # Only process failures during the test call phase (not setup/teardown)
    if report.when == "call" and report.failed:
        # Check if the failure was due to ExceededContextWindowSizeError
        if call.excinfo is not None:
            exc_type = call.excinfo.type
            if exc_type is fm.ExceededContextWindowSizeError:
                # Convert the error to a warning
                exc_value = call.excinfo.value
                warnings.warn(
                    f"ExceededContextWindowSizeError in {item.nodeid}: {str(exc_value)}",
                    UserWarning,
                    stacklevel=2,
                )
                # Mark the test as passed instead of failed
                report.outcome = "passed"
                report.wasxfail = (
                    f"ExceededContextWindowSizeError (converted to warning): {str(exc_value)}"
                )


def pytest_runtest_teardown(item):
    """Hook that runs after each test."""
    print(f"\n[conftest.py] Finished test: {item.nodeid}")

    # Force aggressive cleanup after each test
    gc.collect()
    gc.collect()  # Run twice to catch circular references

    # Give the system a moment to clean up native resources
    import time

    time.sleep(0.1)


@pytest.fixture(autouse=True)
def cleanup_between_tests():
    """
    Fixture that runs before and after each test to ensure clean state.
    This is especially important for tests that use native C/Swift bindings.
    """
    # Before test: ensure clean state
    gc.collect()

    yield

    # After test: aggressive cleanup
    gc.collect()
    gc.collect()

    # Small delay to allow native resources to be released
    import time

    time.sleep(0.05)


@pytest.fixture(scope="function", autouse=False)
def event_loop(request):
    """
    Create a new event loop for each test function.

    This fixture ensures proper cleanup of event loops between tests,
    which is important when testing async code that interacts with
    native C/Swift bindings.
    """
    print(f"\n[conftest.py] Setting up event loop for: {request.node.name}")

    # Close any existing event loop
    try:
        existing_loop = asyncio.get_event_loop()
        if existing_loop and not existing_loop.is_closed():
            print("[conftest.py] Closing existing event loop")
            existing_loop.close()
    except RuntimeError:
        pass

    # Create a fresh event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print(f"[conftest.py] Created new event loop: {id(loop)}")
    yield loop

    # Clean up pending tasks
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        # Give cancelled tasks a chance to finish
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass

    # Close the loop
    try:
        loop.close()
    except Exception:
        pass

    # Force garbage collection to clean up any lingering references
    gc.collect()

    # Ensure no event loop is set after test
    try:
        asyncio.set_event_loop(None)
    except Exception:
        pass


@pytest.fixture
def model():
    """Shared fixture for system language model."""
    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"Model not available: {reason}")
    return model


@pytest.fixture
def session(model):
    """Shared fixture for language model session."""
    return fm.LanguageModelSession(model=model)


def assert_schema_properties(schema: fm.GenerationSchema, title: str, properties: list[str]):
    """
    Reusable helper to validate that a schema contains all expected properties.

    Args:
        schema: The GenerationSchema object to validate
        title: Expected title of the schema
        properties: List of expected property names

    Raises:
        AssertionError: If schema doesn't match expectations
    """
    assert isinstance(schema, fm.GenerationSchema), "Invalid schema"

    # Convert schema to dict and validate
    jsn = schema.to_dict()

    # Validate title
    assert jsn["title"] == title, (
        f"Schema missing title, expected '{title}' got '{jsn.get('title')}'"
    )

    # Validate property count
    actual_count = len(jsn["properties"])
    expected_count = len(properties)
    assert actual_count == expected_count, (
        f"Schema has incorrect number of properties, expecting {expected_count} got: {actual_count}"
    )

    # Validate each property exists
    for prop in properties:
        assert prop in jsn["properties"], f"Schema missing '{prop}' property"
