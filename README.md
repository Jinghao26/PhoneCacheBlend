# PhoneCacheBlend

On-device RAG with KV chunk reuse for iPhone (llama.cpp + Metal).

## Layout

```
PhoneCacheBlend/
  plan.txt              Production journal
  llama.cpp -> ../llama.cpp   Symlink to upstream llama.cpp (do not duplicate)
  app/                  iOS app (forked from llama.swiftui)
  cache/                Chunk KV blobs at runtime (device: Documents/chunks/)
  docs/MODIFY.md        What files to change per phase
```

## Build (first time)

1. Build the XCFramework from llama.cpp:

```bash
cd ~/Desktop/llama.cpp
./build-xcframework.sh
```

2. Open the Xcode project:

```bash
open ~/Desktop/PhoneCacheBlend/app/llama.swiftui.xcodeproj
```

3. Add Qwen2.5-1.5B Q4_K_M GGUF to the app (via model download UI or Documents/).

4. Build & run on iPhone 16 (Metal). Simulator uses CPU only.

## Model

- **Qwen2.5-1.5B-Instruct** GGUF **Q4_K_M**
- Example: [Hugging Face Qwen2.5 GGUF collections](https://huggingface.co/Qwen)

## v0.1 goal

Simple KV reuse by `chunk_id`: save chunk state → load → stitch → prefill question only.

See `plan.txt` for full journal.
