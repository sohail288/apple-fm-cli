#!/usr/bin/env bash
set -e

mkdir -p "$HOME/.local/bin"

echo "Installing apple-fm-cli using uv tool..."
uv tool install . --force

echo "Installation complete!"
echo "Ensure that ~/.local/bin is in your PATH to use 'apple-fm-cli'."
