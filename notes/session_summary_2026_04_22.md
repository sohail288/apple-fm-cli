# Session Summary - April 22, 2026

## What We Achieved
- **Native Embedding Support**: Implemented a bridge to Apple's `NaturalLanguage` framework (`NLEmbedding`). We now have a truly native, local 512-D sentence embedding feature in `apple_fm_sdk`.
- **RAG Demonstration**: Created a self-contained example in `examples/rag_demo/` that uses the new native embeddings with Qdrant and Apple Intelligence for local retrieval-augmented generation.
- **CI/CD Consolidation**: Unified the lint, test, and publish workflows into a single `workflow.yml` with explicit dependencies.
- **Version Release**: Bumped the project to `v0.2.0`.
- **Binding Recovery (same day)**: Diagnosed and fixed the “missing `FMSystemLanguageModelCreate`” issue; see **Root cause & resolution** below.

## What We Learned
- **macOS 26 API Shifts**: The `FoundationModels` framework is evolving rapidly. `tokenCount(for:)` has moved between sync and async shapes across SDKs, and property names like `contextSize` vs `maxContextSize` vary across seeds. Compatibility code must avoid **linking** a Swift entry point you cannot rely on at load time.
- **Compiler and linking**: A direct `await self.tokenCount(for:)` in the bridge (even behind `#available`) can still record a dependency on the **async** `tokenCount` symbol. If that symbol is missing from the `FoundationModels` binary on the running OS, **`dlopen` of the custom dylib fails entirely**—before Python ever resolves C names.
- **`ctypes` / Darwin**: `hasattr(cdll, "FMSystemLanguageModelCreate")` failed not because of a leading `_` in `dlsym`, but because Python had bound the **wrong** library: when the **bundled** dylib failed to open, the generic `load_library("…")` search fell through to **Apple’s** `FoundationModels.framework`, which does not export our `@_cdecl`bridge symbols.
- **Loader behavior**: Generic library names such as `FoundationModels` or `apple_fm_bridge` are unsafe: they can match the system framework. The bridge must be loaded by **path** to the dylib shipped under `apple_fm_sdk/lib/`.

## Root Cause & Resolution (April 22 follow-up)
1. **Failed `dlopen` of the bridge**: The Swift build targeted macOS 26.4+ and the compatibility layer called the async `tokenCount` API, pulling in a Swift async symbol that was not present for the on-disk framework version. The bundled `lib…dylib` then failed to load.
2. **Silent wrong library**: After that failure, ctypes loaded **`/System/Library/…/FoundationModels.framework`**, so none of the C entry points (`FMSystemLanguageModelCreate`, `FMGetSentenceEmbedding`, etc.) appeared on the `CDLL` object.
3. **Fixes applied**:
   - **Swift** (`FoundationModelsCBindings.swift`): Stopped calling `tokenCount(for:)` directly; use the existing protocol / `contextSize` compatibility path only, so the bridge does not depend on that async symbol.
   - **Python** (`_ctypes_bindings.py`): Load only **`libapple_fm_bridge.dylib`** or the legacy filename **`libFoundationModels.dylib`** by absolute path under the package. No fallback to the system framework; clear `ImportError` if the bridge cannot be opened.
   - **CI** (`workflow.yml`): Copy the built product with `swift build -c release --show-bin-path` and `libapple_fm_bridge.dylib` into `apple_fm_sdk/lib/`.
   - **Package / README**: `Package.swift` minimum platforms remain in the **macOS 26** family (the module uses 26-gated `FoundationModels` APIs); docs note both dylib names.

## Verification
- `tests/test_embeddings.py` exercises real **`FMGetSentenceEmbedding`** (512-d floats, distinct texts, idempotence).
- **E2E** tests (`-m e2e`) run the real CLI against on-device Foundation Models when available.
- `otool -L` on the shipped bridge shows linkage to `FoundationModels.framework` and `NaturalLanguage.framework` (NLEmbedding), not a self-contained stub.

## Obsolete / incorrect hypotheses (archived)
- **Leading underscore in `dlsym`**: Not the root cause; symbols like `_FMSystemLanguageModelCreate` are present in the **correct** dylib; the issue was loading the **wrong** binary.
- **CLI `arguments_schema`**: `apple_fm_cli` tools already expose `arguments_schema` as a property; this was not the source of the native binding failure.

## Follow-ups (optional)
- If `tokenCount` for short strings still looks like a context cap (e.g. 4096), audit **`compatTokenCount`** vs. real `tokenCount` behavior on the minimum supported macOS.
- Regenerate or thin **`notes/pypi_releases_and_packaging.md`** references if they still say only `libFoundationModels.dylib` without `libapple_fm_bridge.dylib`.
