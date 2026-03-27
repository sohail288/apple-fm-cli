import asyncio
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock, patch

from apple_fm_cli import BashTool, GoogleSearchTool

@pytest.mark.asyncio
async def test_bash_tool_ls():
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
            "ls",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

@pytest.mark.asyncio
async def test_bash_tool_error():
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
async def test_google_search_tool():
    tool = GoogleSearchTool()
    
    @dataclass
    class MockArgs:
        query: str
    
    args = MockArgs(query="test query")
    
    mock_html = """
    <html>
        <tr class="result-snippet"><td class="result-snippet">Result 1 snippet</td></tr>
        <tr class="result-snippet"><td class="result-snippet">Result 2 snippet</td></tr>
    </html>
    """
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = mock_html.encode("utf-8")
        mock_urlopen.return_value = mock_response
        
        # GoogleSearchTool uses run_in_executor, so we need to mock that or allow it to run
        # Since we're patching urlopen, run_in_executor will call the mock.
        
        result = await tool.call(args)
        
        assert "Result 1 snippet" in result
        assert "Result 2 snippet" in result

@pytest.mark.asyncio
async def test_google_search_tool_no_results():
    tool = GoogleSearchTool()
    
    @dataclass
    class MockArgs:
        query: str
    
    args = MockArgs(query="no results")
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html>No results here</html>"
        mock_urlopen.return_value = mock_response
        
        result = await tool.call(args)
        
        assert result == "No results found."
