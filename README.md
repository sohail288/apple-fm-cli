# apple-fm-cli

A small **command-line interface** and **local HTTP server** that drive Apple’s on-device **Foundation Models** (Apple Intelligence) from Python. The repo bundles a ctypes-based **`apple_fm_sdk`** wrapper around the system model plus **`apple_fm_cli`**, which adds prompting, optional tools, JSON-schema-guided output, and an OpenAI-shaped API for clients like Codex.

## Requirements

- **macOS** with Foundation Models available (the SDK checks availability and reports a reason if not).
- **Python ≥ 3.14**

## Install

**From PyPI** (macOS, Python 3.14+):

```bash
pip install apple-fm-cli
# or: uv pip install apple-fm-cli
```

**From a git checkout** (editable):

```bash
pip install -e .
# or: uv pip install -e .
```

Entry point: `apple-fm-cli`.

The published wheel includes a prebuilt native bridge (Apple silicon / arm64), typically **`libapple_fm_bridge.dylib`** (or the legacy name **`libFoundationModels.dylib`**). To rebuild it after changing `foundation-models-c`, run `swift build -c release` in `src/apple_fm_sdk/foundation-models-c` and copy `.build/*/release/libapple_fm_bridge.dylib` to `src/apple_fm_sdk/lib/` before building distributions.

### Publishing to PyPI (maintainers)

**Automated (recommended):** [`.github/workflows/publish.yml`](.github/workflows/publish.yml) runs when you **push a version tag** `v*` (for example `v0.1.3`). It checks that the tag matches **`version` in `pyproject.toml`**, builds, uploads to PyPI, then **creates a GitHub Release** (with generated notes) if one does not already exist. One-time: configure [PyPI trusted publishing](https://pypi.org/manage/account/publishing/) for workflow **`publish.yml`**.

Typical release steps:

1. Bump **`version`** in **`pyproject.toml`** and run **`uv lock`** if you track **`uv.lock`** in git.
2. Rebuild and commit **`src/apple_fm_sdk/lib/libapple_fm_bridge.dylib`** (or **`libFoundationModels.dylib`**) if the Swift bridge changed.
3. Commit and push to **`main`**, then tag and push the tag:

   ```bash
   git tag v0.1.3
   git push origin v0.1.3
   ```

**Manual dispatch:** **Actions → Publish to PyPI → Run workflow** still works for the current branch (no tag check); use sparingly.

**Manual upload:** `uv run --with build python -m build` then `uv run --with twine twine upload dist/*` using an API token. Prefer **trusted publishing** in CI over storing long-lived tokens in `~/.pypirc` or GitHub secrets.

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

## Local embedding benchmark

From the repo root, measure end-to-end latency for **512-d** English sentence embeddings (native `NLEmbedding` or the HTTP `POST /v1/embeddings` route on the local server):

```bash
uv run python scripts/benchmark_embeddings.py
uv run python scripts/benchmark_embeddings.py -n 100 -w 5
uv run python scripts/benchmark_embeddings.py --json
# With the server: apple-fm-cli server --port 8000
uv run python scripts/benchmark_embeddings.py --mode http --base-url http://127.0.0.1:8000
```

Use `--batch` to time multi-string requests (HTTP sends one `POST` per iteration; native runs a tight loop). Results print throughput, dimension, and latency percentiles.

## Server

Starts a **FastAPI** app that mimics parts of the **OpenAI Chat Completions**, **Embeddings**, and **Responses** APIs (including SSE for streaming), so tools that expect those endpoints can point at your machine instead of a cloud provider. Large agent system prompts are truncated heuristically to fit smaller local context windows.

```bash
apple-fm-cli server --host 0.0.0.0 --port 8000
```

- `POST /v1/chat/completions`
- `POST /v1/embeddings` — on-device **512-dimensional** English sentence embeddings (`NaturalLanguage` / `NLEmbedding`); the `model` field is accepted and echoed (OpenAI compatibility) but does not select a different backend.
- `POST /v1/responses`

### Codex

1. Start the server (see above), e.g. on port `8000`.
2. Add a provider and profile to **`~/.codex/config.toml`**:

   ```toml
   [model_providers.apple]
   name = "apple"
   base_url = "http://localhost:8000/v1"
   env_key = "OPENAI_API_KEY"

   [profiles.apple]
   model = "fm"
   model_provider = "apple"
   model_context_window = 4096
   ```

3. Run Codex with that profile:

   ```bash
   codex -p apple
   ```

`env_key` is the environment variable Codex uses for the bearer token. The local server does not need a real OpenAI key; set `OPENAI_API_KEY` to any non-empty placeholder if your Codex build requires it to be present.

### Other agent harnesses

Anything that can target an **OpenAI-compatible** HTTP API (Chat Completions and/or Responses, including SSE) can point **`base_url`** at `http://<host>:<port>/v1` and use a **model id** string of your choice—the server echoes the requested model name. Prefer a **context window** that matches what the on-device model can handle (4096 is a reasonable default for local sessions). If the client insists on an API key, keep using a dummy value in the configured env var unless you add auth in front of the server yourself.

## Layout

| Path | Role |
|------|------|
| `src/apple_fm_sdk/` | Session, tools, guided generation, tokenizer, native bridge bindings |
| `src/apple_fm_cli/` | `query` / `server`, built-in tools, schema → `Generable` helpers |
| `scripts/` | Helpers, e.g. `benchmark_embeddings.py` for local latency/throughput |
| `notes/` | Design notes (Responses SSE lifecycle, native bridge, e2e testing) |

## Licensing

`apple_fm_sdk` source files carry Apple Inc. copyright headers; refer to any accompanying license text distributed with that SDK.
