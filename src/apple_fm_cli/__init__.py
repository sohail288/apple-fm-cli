from typing import Any

__all__ = [
    "BashTool",
    "GoogleSearchTool",
    "create_dynamic_dataclass",
    "main",
]


def main() -> None:
    from .cli import main as cli_main

    cli_main()


def __getattr__(name: str) -> Any:
    if name in {"BashTool", "GoogleSearchTool", "create_dynamic_dataclass"}:
        from . import cli

        return getattr(cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
