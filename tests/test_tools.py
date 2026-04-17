import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

try:
    from apple_fm_cli import BashTool, GoogleSearchTool
except ImportError as error:
    pytest.skip(
        f"Foundation Models C bindings unavailable: {error}",
        allow_module_level=True,
    )


@pytest.mark.asyncio
async def test_bash_tool_ls() -> None:
    tool = BashTool()

    # Mocking arguments object
    @dataclass
    class MockArgs:
        command: str

    args = MockArgs(command="ls")

    # We want to verify it calls create_subprocess_shell
    with patch("asyncio.create_subprocess_shell") as mock_subproc:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"file1\nfile2", b"")
        mock_subproc.return_value = mock_process

        result = await tool.call(args)

        assert "file1" in result
        assert "file2" in result
        mock_subproc.assert_called_once_with(
            "ls", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )


@pytest.mark.asyncio
async def test_bash_tool_error() -> None:
    tool = BashTool()

    @dataclass
    class MockArgs:
        command: str

    args = MockArgs(command="false")

    with patch("asyncio.create_subprocess_shell") as mock_subproc:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error message")
        mock_subproc.return_value = mock_process

        result = await tool.call(args)

        assert "STDERR:" in result
        assert "error message" in result


@pytest.mark.asyncio
async def test_google_search_tool(httpx_mock: Any) -> None:
    tool = GoogleSearchTool()

    @dataclass
    class MockArgs:
        query: str

    args = MockArgs(query="test query")

    mock_html = """
    <html>
        <tr class="result-snippet">
            <td class="result-snippet">Result 1 snippet</td>
        </tr>
        <tr>
            <td><a class="result-link" href="https://example.com/1">Result 1 link</a></td>
        </tr>
    </html>
    """

    # Mock the DuckDuckGo search request
    httpx_mock.add_response(
        url="https://lite.duckduckgo.com/lite/",
        method="POST",
        text=mock_html,
    )

    # Mock the page fetch request
    httpx_mock.add_response(
        url="https://example.com/1",
        method="GET",
        text="<html><body>Page content here</body></html>",
    )

    result = await tool.call(args)

    assert "Result 1 snippet" in result
    assert "Page content here" in result
    assert "https://example.com/1" in result


@pytest.mark.asyncio
async def test_google_search_tool_no_results(httpx_mock: Any) -> None:
    tool = GoogleSearchTool()

    @dataclass
    class MockArgs:
        query: str

    args = MockArgs(query="no results")

    httpx_mock.add_response(
        url="https://lite.duckduckgo.com/lite/",
        method="POST",
        text="<html>No results here</html>",
    )

    result = await tool.call(args)

    assert result == "No results found."
