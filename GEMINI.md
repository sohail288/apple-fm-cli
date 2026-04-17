# Project Mandates: apple-fm-cli

## Technical Integrity & Native Implementation
- **No Stubbing or Mimicking:** All functionality described in the Apple Foundation Models documentation must be implemented via actual native calls to the `FoundationModels` framework. Do not use Python-side proxies (e.g., `tiktoken` for token counting) if a corresponding native API exists or can be bridged.
- **Native Bridge Extension:** If a native API is missing from the current Python SDK C-bindings, you MUST extend the Swift bridge (`foundation-models-c`) and re-compile the library to expose the true native functionality.
- **Transparency:** Clearly document which parts of the SDK are bridged and ensure that the implementation matches the Swift framework's behavior and performance characteristics.
- **Hardware Awareness:** Maintain the skip logic for CI/CD environments that lack physical Neural Engine support, but ensure the logic being skipped is the actual native code path.
