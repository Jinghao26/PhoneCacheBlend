# Files to modify — PhoneCacheBlend

Fork layout: `app/` is a copy of `llama.cpp/examples/llama.swiftui`.
Upstream C++ is **not** copied; use symlink `PhoneCacheBlend/llama.cpp`.

Only modify llama.cpp C++ when you need APIs that Swift cannot reach (e.g. raw
KV export). For v0.1, stay in Swift + public `llama.h` state APIs.

---

## Project map

| Path | Role |
|------|------|
| `app/llama.cpp.swift/LibLlama.swift` | **Core** — llama.cpp wrapper: load model, prefill, decode, KV clear |
| `app/llama.swiftui/Models/LlamaState.swift` | App state: model list, `complete()`, timing logs |
| `app/llama.swiftui/UI/ContentView.swift` | Main UI — **RAG UI goes here (Phase A)** |
| `app/llama.swiftui/UI/*.swift` | Model download / input helpers |
| `app/llama.swiftui.xcodeproj/project.pbxproj` | Xcode paths (points to `../llama.cpp/build-apple/`) |
| `llama.cpp/build-xcframework.sh` | Build once; produces `build-apple/llama.xcframework` |
| `cache/` | Dev placeholder; on device use `Documents/chunks/` |

---

## Phase A — Baseline RAG (modify now)

### 1. `LibLlama.swift` — inference settings

- [ ] `create_context`: enable mmap via `model_params.use_mmap`
- [ ] `ctx_params.n_ctx = 2048` (already set)
- [ ] `n_gpu_layers` for device (already Metal on device)
- [ ] Add `n_seq_max` / batch seq support if multi-sequence needed later
- [ ] Add **TTFT logging** in `completion_init` (timestamp around `llama_decode`)
- [ ] Add `prefill(text: String, startPos: Int32)` for partial prefill at `n_past`

### 2. `LlamaState.swift` — model + RAG orchestration

- [ ] Replace `defaultModels` with **Qwen2.5-1.5B-Instruct Q4_K_M** URL
- [ ] Add `completeRag(passages: [String], question: String)` — concat → full prefill
- [ ] Log: TTFT, tokens prefilled, mode (`full` vs `cached`)

### 3. `ContentView.swift` — UI

- [ ] Separate fields: **Passages** (multi-line) + **Question**
- [ ] Button: **Ask (full prefill)** — baseline
- [ ] Show TTFT in message log

### 4. New files (create in Phase A/B)

| New file | Purpose |
|----------|---------|
| `app/llama.swiftui/Models/ChunkStore.swift` | `chunk_id = SHA256(text)`, paths under Documents/chunks/ |
| `app/llama.swiftui/Models/RagPrompt.swift` | Build `[system + chunks + question]` string |
| `app/llama.cpp.swift/ChunkKV.swift` | `saveChunkState` / `loadChunkState` / `stitch` (Phase B/C) |

Add new Swift files to `project.pbxproj` in Xcode (File → Add Files).

---

## Phase B — Chunk ID save/load (`LibLlama.swift` + `ChunkKV.swift`)

Use public C API (already in xcframework):

```c
llama_state_seq_save_file(ctx, path, seq_id, tokens, n_tokens)
llama_state_seq_load_file(ctx, path, dest_seq_id, tokens_out, ...)
llama_memory_seq_cp(mem, src, dst, p0, p1)
llama_get_memory(ctx)
```

- [ ] `prefillChunk(text)` → tokens, decode seq 0, save `.bin` + metadata `.json`
- [ ] `loadChunk(chunkId)` → restore into scratch `seq_id = 1`
- [ ] `clearMemory()` between experiments

**Do not modify** llama.cpp C++ unless save/load fails for Qwen.

---

## Phase C — Stitch by chunk ID list

- [ ] `stitch(chunkIds: [String])` — copy seq 1 → seq 0 at increasing positions
- [ ] `completeWithCache(chunkIds, question)` — stitch then prefill question only
- [ ] UI: show cache hit/miss per chunk

---

## Phase D — Optional C++ changes (only if needed)

| llama.cpp file | When |
|----------------|------|
| `src/llama-kv-cache.cpp` | Custom stitch, position fixups |
| `include/llama.h` | New export API for per-layer K/V |
| Attention / graph sources | CacheBlend fusion (v2, not v0.1) |

**v0.1 rule:** avoid C++ fork; use `llama_state_seq_*` + `llama_memory_seq_cp`.

---

## What NOT to modify

- Entire `llama.cpp` tree (use symlink, pull upstream updates)
- `vllm` / CacheBlend Python repos (reference only)
- Xcode generated `build/` folders inside llama.cpp

---

## Build checklist

```bash
# 1. XCFramework (from Desktop)
cd ~/Desktop/llama.cpp && ./build-xcframework.sh

# 2. Open app
open ~/Desktop/PhoneCacheBlend/app/llama.swiftui.xcodeproj
```

If Xcode cannot find `llama.xcframework`, run step 1 first.
