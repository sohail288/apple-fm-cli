import json
import os
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

try:
    import apple_fm_sdk as fm
except ImportError as error:
    pytest.skip(
        f"Foundation Models C bindings unavailable: {error}",
        allow_module_level=True,
    )


def run_cli(installed_cli: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [str(installed_cli), *args],
        capture_output=True,
        text=True,
        check=True,
    )  # noqa: S603, S607


@pytest.fixture(scope="session", autouse=True)
def check_fm_available() -> None:
    """Skip all E2E tests if Foundation Models are not available."""
    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"Foundation Models not available: {reason}")


@pytest.fixture(scope="session")
def installed_cli() -> Generator[Path]:
    """
    Fixture to install the CLI into a temporary directory and return the path to the executable.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        # Install the tool into the temporary directory using uv
        venv_dir = tmp_path / ".venv"
        subprocess.run(["uv", "venv", str(venv_dir)], check=True, capture_output=True)  # noqa: S603, S607

        # Determine the python executable in the venv
        if os.name == "nt":
            cli_exe = venv_dir / "Scripts" / "apple-fm-cli"
        else:
            cli_exe = venv_dir / "bin" / "apple-fm-cli"

        # Install the current project into the venv
        subprocess.run(  # noqa: S603, S607
            ["uv", "pip", "install", ".", "--python", str(venv_dir)],  # noqa: S607
            check=True,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
        )  # noqa: S603, S607

        yield cli_exe


@pytest.mark.e2e
def test_cli_basic_query(installed_cli: Path) -> None:
    # Given
    query = "What is 2 + 2?"

    # When
    result = run_cli(installed_cli, "-q", query)

    # Then
    assert "4" in result.stdout


@pytest.mark.e2e
def test_cli_json_output(installed_cli: Path) -> None:
    # Given
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    prompt = "Generate a cat"

    # When
    result = run_cli(
        installed_cli,
        "-q",
        prompt,
        "--output",
        "json",
        "--output-schema",
        json.dumps(schema),
    )

    # Then
    data = json.loads(result.stdout)
    assert isinstance(data.get("name"), str)
    assert isinstance(data.get("age"), int)


@pytest.mark.e2e
def test_cli_bash_tool(installed_cli: Path) -> None:
    # Given
    prompt = "Read the file named pyproject.toml and tell me what is the 'name' of the project."
    tool_name = "bash"

    # When
    result = run_cli(installed_cli, "-q", prompt, "--tools", tool_name)

    # Then
    assert "apple-fm-cli" in result.stdout.lower()
