import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def installed_cli() -> Generator[Path, None, None]:
    """
    Fixture to install the CLI into a temporary directory and return the path to the executable.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        # Install the tool into the temporary directory using uv
        venv_dir = tmp_path / ".venv"
        subprocess.run(
            ["uv", "venv", str(venv_dir)],
            check=True,
            capture_output=True
        )
        
        # Determine the python executable in the venv
        if os.name == "nt":
            cli_exe = venv_dir / "Scripts" / "apple-fm-cli"
        else:
            cli_exe = venv_dir / "bin" / "apple-fm-cli"

        # Install the current project into the venv
        subprocess.run(
            ["uv", "pip", "install", ".", "--python", str(venv_dir)],
            check=True,
            cwd=Path(__file__).parent.parent,
            capture_output=True
        )

        yield cli_exe


@pytest.mark.e2e
def test_cli_basic_query(installed_cli: Path) -> None:
    """Test a basic text query."""
    result = subprocess.run(
        [str(installed_cli), "-q", "What is 2 + 2?"],
        capture_output=True,
        text=True,
        check=True
    )
    assert "4" in result.stdout


@pytest.mark.e2e
def test_cli_json_output(installed_cli: Path) -> None:
    """Test structured JSON output with a schema."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    
    result = subprocess.run(
        [
            str(installed_cli),
            "-q", "Generate a cat",
            "--output", "json",
            "--output-schema", json.dumps(schema)
        ],
        capture_output=True,
        text=True,
        check=True
    )
    
    data = json.loads(result.stdout)
    assert isinstance(data.get("name"), str)
    assert isinstance(data.get("age"), int)


@pytest.mark.e2e
def test_cli_bash_tool(installed_cli: Path) -> None:
    """Test tool usage (bash)."""
    # We ask the model to read a specific file using the bash tool.
    # We include an explicit instruction to use the tool.
    result = subprocess.run(
        [
            str(installed_cli),
            "-q", "Read the file named pyproject.toml and tell me what is the 'name' of the project.",
            "--tools", "bash"
        ],
        capture_output=True,
        text=True,
        check=True
    )
    
    assert "apple-fm-cli" in result.stdout.lower()
