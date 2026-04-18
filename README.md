# apple-fm-cli

A small **command-line interface** and **local HTTP server** that drive Apple’s on-device **Foundation Models** (Apple Intelligence) from Python. The repo bundles a ctypes-based **`apple_fm_sdk`** wrapper around the system model plus **`apple_fm_cli`**, which adds prompting, optional tools, JSON-schema-guided output, and an OpenAI-shaped API for clients like Codex.

## Requirements

- **macOS** with Foundation Models available (the SDK checks availability and reports a reason if not).
- **Python ≥ 3.14**

## Install

```bash
pip install -e .
# or: uv pip install -e .
```

Entry point: `apple-fm-cli`.

## CLI

**Query the model** (plain text or structured JSON):

```bash
apple-fm-cli query "Summarize this idea in one sentence."
apple-fm-cli query --format json --schema '{"type":"object","properties":{"title":{"type":"string"}}}' "Name this topic."
```

**Optional tools** (comma-separated): `bash` (local shell), `google_search` (DuckDuckGo lite + page fetch).

```bash
apple-fm-cli query --tools bash,google_search "What’s in README.md in the cwd?"
```

Legacy-style flags are still accepted: `-q` / `--query`, `--output`, `--output-schema`.

## Server

Starts a **FastAPI** app that mimics parts of the **OpenAI Chat Completions** and **Responses** APIs (including SSE for streaming), so tools that expect those endpoints can point at your machine instead of a cloud provider. Large agent system prompts are truncated heuristically to fit smaller local context windows.

```bash
apple-fm-cli server --host 0.0.0.0 --port 8000
```

- `POST /v1/chat/completions`
- `POST /v1/responses`

## Layout

| Path | Role |
|------|------|
| `src/apple_fm_sdk/` | Session, tools, guided generation, tokenizer, native bridge bindings |
| `src/apple_fm_cli/` | `query` / `server`, built-in tools, schema → `Generable` helpers |
| `notes/` | Design notes (Responses SSE lifecycle, native bridge, e2e testing) |

## Licensing

`apple_fm_sdk` source files carry Apple Inc. copyright headers; refer to any accompanying license text distributed with that SDK.
