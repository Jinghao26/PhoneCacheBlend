import Foundation
import llama

/// Metrics for Phase A baseline (full prefill RAG).
struct InferenceMetrics: Sendable {
    let mode: String
    let promptTokens: Int
    let prefillMs: Double
    let ttftMs: Double
    let decodeTokens: Int
    let decodeMs: Double
    let decodeTps: Double

    var summary: String {
        String(format: """
            --- Metrics (%@) ---
            Prompt tokens: %d
            Prefill time:  %.1f ms
            TTFT:          %.1f ms
            Decode tokens: %d
            Decode time:   %.1f ms
            Decode speed:  %.1f tok/s
            """,
            mode,
            promptTokens,
            prefillMs,
            ttftMs,
            decodeTokens,
            decodeMs,
            decodeTps
        )
    }
}

/// Phase E: CacheBlend fuse driver mode (maps to `llama_cache_blend_mode`).
enum CacheBlendFuseMode: Sendable {
    case tokenRecompute
    case graph

    var cMode: llama_cache_blend_mode {
        switch self {
        case .tokenRecompute: return LLAMA_CACHE_BLEND_MODE_TOKEN_RECOMPUTE
        case .graph: return LLAMA_CACHE_BLEND_MODE_GRAPH
        }
    }

    var label: String {
        switch self {
        case .tokenRecompute: return "TOKEN_RECOMPUTE"
        case .graph: return "GRAPH"
        }
    }
}

enum LlamaError: Error, LocalizedError {
    case couldNotInitializeContext
    case chunkPrefillFailed(tokenCount: Int)
    case chunkSaveFailed(path: String)
    case couldNotLoadCachedChunk
    case hkvdExportFailed(Int32)
    case hkvdComputeFailed(Int32)
    case hkvdShapeMismatch(oldTokenCount: Int, newTokenCount: Int)
    case cacheBlendFuseFailed(Int32)
    case promptTooLong(prompt: Int, maxGen: Int, nCtx: Int)
    case samplingLogitsUnavailable

    var errorDescription: String? {
        switch self {
        case .couldNotInitializeContext:
            return "could not initialize llama context"
        case .chunkPrefillFailed(let tokenCount):
            return "chunk KV prefill failed (\(tokenCount) tokens; GPU decode error)"
        case .chunkSaveFailed(let path):
            return "chunk KV save failed (\(URL(fileURLWithPath: path).lastPathComponent))"
        case .couldNotLoadCachedChunk:
            return "could not load cached chunk"
        case .promptTooLong(let prompt, let maxGen, let nCtx):
            return "prompt (\(prompt)) + max_gen (\(maxGen)) exceeds n_ctx (\(nCtx))"
        case .samplingLogitsUnavailable:
            return "logits unavailable for sampling (prefill or fuse may have failed)"
        case .cacheBlendFuseFailed(let code):
            switch code {
            case -1: return "cache blend fuse: null argument"
            case -2: return "cache blend fuse: invalid params"
            case -3: return "cache blend fuse: invalid check layer"
            case -4: return "cache blend fuse: KV state error"
            case -5: return "cache blend fuse: HKVD failed"
            case -6: return "cache blend fuse: decode failed"
            case -10: return "cache blend fuse: not implemented"
            default: return "cache blend fuse failed (\(code))"
            }
        case .hkvdExportFailed(let code):
            switch code {
            case -1: return "hkvd export: null argument"
            case -2: return "hkvd export: no model or empty sequence state"
            case -3: return "hkvd export: state read size mismatch"
            case -4: return "hkvd export: KV layer extract failed (state parse)"
            case -5: return "hkvd export: sequence has no KV cells"
            case -6: return "hkvd export: output buffer too small"
            case -7: return "hkvd export: unsupported V quant type"
            case -8: return "hkvd export: invalid layer head shape"
            case -9: return "hkvd export: memory type not supported (need KV cache)"
            case -10: return "hkvd export: missing KV cell for sequence position"
            default: return "hkvd export failed (\(code))"
            }
        case .hkvdComputeFailed(let code):
            return "hkvd compute failed (\(code))"
        case .hkvdShapeMismatch(let oldCount, let newCount):
            return "hkvd v_old/v_new mismatch (stitched \(oldCount) tokens vs fresh \(newCount))"
        }
    }
}

/// One layer's V cache exported as float32 `[n_tokens][n_kv_heads][head_dim]`.
struct LayerVExport: Sendable {
    let values: [Float]
    let nTokens: UInt32
    let nKvHeads: UInt32
    let headDim: UInt32
}

/// Phase E: result of `llama_cache_blend_fuse` on stitched KV.
struct CacheBlendFuseResult: Sendable {
    let mode: CacheBlendFuseMode
    let checkLayer: Int32
    let nTokens: UInt32
    let suffixLen: UInt32
    let recompRatio: Float
    let indices: [UInt32]
    let fuseMs: Double

    var topkPrefixCount: UInt32 {
        guard indices.count >= suffixLen else { return 0 }
        return UInt32(indices.count) - suffixLen
    }

    var summary: String {
        let idxPreview = indices.prefix(16).map(String.init).joined(separator: ", ")
        let more = indices.count > 16 ? ", …" : ""
        return String(
            format: """
            --- CacheBlend fuse (%@, layer %d) ---
            Tokens: %u (suffix %u, top-K prefix %u @ ratio %.2f)
            Fuse time: %.1f ms
            |imp| = %u: [%@%@]
            """,
            mode.label,
            checkLayer,
            nTokens,
            suffixLen,
            topkPrefixCount,
            recompRatio,
            fuseMs,
            indices.count,
            idxPreview,
            more
        )
    }
}

/// Phase D: CacheBlend-style HKVD probe result.
struct HkvdProbeResult: Sendable {
    let checkLayer: Int32
    let nTokens: UInt32
    let nKvHeads: UInt32
    let headDim: UInt32
    let suffixLen: UInt32
    let recompRatio: Float
    let indices: [UInt32]
    let exportOldMs: Double
    let fullPrefillMs: Double
    let exportNewMs: Double
    let computeMs: Double

    var topkPrefixCount: UInt32 {
        guard indices.count >= suffixLen else { return 0 }
        return UInt32(indices.count) - suffixLen
    }

    var summary: String {
        let idxPreview = indices.prefix(16).map(String.init).joined(separator: ", ")
        let more = indices.count > 16 ? ", …" : ""
        return String(
            format: """
            --- HKVD probe (layer %d) ---
            Tokens: %u (suffix %u, top-K prefix %u @ ratio %.2f)
            Heads: %u × dim %u
            Export old V:  %.1f ms
            Full prefill:  %.1f ms
            Export new V:  %.1f ms
            Compute HKVD:  %.1f ms
            Indices (%u): [%@%@]
            """,
            checkLayer,
            nTokens,
            suffixLen,
            topkPrefixCount,
            recompRatio,
            nKvHeads,
            headDim,
            exportOldMs,
            fullPrefillMs,
            exportNewMs,
            computeMs,
            indices.count,
            idxPreview,
            more
        )
    }
}

/// Cached system prefix for Phase C stitch (BOS + `systemPrefix` KV blob).
struct PrefixStitchItem: Sendable {
    let chunkId: String
    let binPath: String
    let nTokens: Int
    let ramState: Data?
}

/// One cached chunk for Phase C stitch (disk path + optional RAM snapshot).
struct ChunkStitchItem: Sendable {
    let chunkId: String
    /// Fresh-prefilled at stitch time, e.g. `"[1] "`. Body KV in `binPath` excludes this label.
    let labelText: String
    let binPath: String
    let nTokens: Int
    let ramState: Data?
    /// RAM snapshot for `labelText` when label KV cache is warm (merge at stitch instead of GPU).
    let labelRamState: Data?
    let labelNTokens: Int
}

/// Per-component stitch wall times (ms), each including its GPU/Metal fence where applicable.
struct StitchTimingBreakdown: Sendable {
    var clearMs: Double = 0
    /// `llama_state_seq_set_data` / load + unpack (incl. post-op fence).
    var deserializeRamMs: Double = 0
    var deserializeDiskMs: Double = 0
    /// `seq_add` + `seq_cp` + `seq_rm` (incl. post-op fence).
    var memcpyMs: Double = 0
    /// GPU prefill for `[n] ` labels (incl. fence per label batch).
    var labelGpuMs: Double = 0
    /// KV merge for cached `[n] ` labels (deserialize + seq_cp, incl. fence).
    var labelCacheMs: Double = 0
    var labelTokens: Int = 0
    /// GPU prefill for question suffix during stitch (incl. fence).
    var questionGpuMs: Double = 0
    /// GPU prefill for question suffix after fuse (incl. fence).
    var questionPostFuseMs: Double = 0
    var questionTokens: Int = 0
    /// Fresh system-prefix GPU prefill when prefix cache unavailable (incl. fence).
    var prefixGpuMs: Double = 0
    var prefixTokens: Int = 0
    /// Post-disk-load RAM snapshot (`captureSequenceState`) for hot cache warm.
    var snapshotWarmMs: Double = 0
    var mergeCount: Int = 0
    var ramHits: Int = 0
    var diskLoads: Int = 0

    var deserializeMs: Double { deserializeRamMs + deserializeDiskMs }

    var coreMs: Double {
        clearMs + deserializeMs + memcpyMs + labelGpuMs + labelCacheMs + questionGpuMs + prefixGpuMs
    }

    var accountedMs: Double { coreMs + snapshotWarmMs }

    func formattedLog(stitchTotalMs: Double) -> String {
        var lines: [String] = []
        lines.append("--- Stitch breakdown (times incl. Metal fence) ---")
        lines.append(String(format: "  Clear:           %.1f ms", clearMs))
        lines.append(String(
            format: "  Deserialize:     %.1f ms  (RAM %.1f + disk %.1f, %d merges)",
            deserializeMs, deserializeRamMs, deserializeDiskMs, mergeCount
        ))
        lines.append(String(format: "  Memcpy (seq_cp): %.1f ms  (seq_add/cp/rm + fence)", memcpyMs))
        if prefixGpuMs > 0 {
            lines.append(String(
                format: "  Prefix GPU:      %.1f ms  (%d tok)",
                prefixGpuMs, prefixTokens
            ))
        }
        if labelCacheMs > 0 {
            lines.append(String(
                format: "  Label cache:     %.1f ms  (%d tok, KV merge)",
                labelCacheMs, labelTokens
            ))
        }
        if labelGpuMs > 0 {
            lines.append(String(
                format: "  Label GPU:       %.1f ms  (%d tok)",
                labelGpuMs, labelTokens
            ))
        } else if labelCacheMs == 0 {
            lines.append(String(
                format: "  Label GPU:       %.1f ms  (%d tok)",
                labelGpuMs, labelTokens
            ))
        }
        lines.append(String(
            format: "  Question GPU:    %.1f ms  (%d tok)",
            questionGpuMs, questionTokens
        ))
        if questionPostFuseMs > 0 {
            lines.append(String(
                format: "  Question post-fuse: %.1f ms  (%d tok)",
                questionPostFuseMs, questionTokens
            ))
        }
        if snapshotWarmMs > 0 {
            lines.append(String(format: "  Snapshot warm:   %.1f ms  (disk→RAM capture)", snapshotWarmMs))
        }
        lines.append(String(format: "  ─────────────────────────"))
        lines.append(String(
            format: "  Sum core:        %.1f ms  (clear+deser+memcpy+label+question%@)",
            coreMs,
            prefixGpuMs > 0 ? "+prefix" : ""
        ))
        lines.append(String(
            format: "  Sum all:         %.1f ms  (residual %+.1f ms vs stitch %.1f ms)",
            accountedMs, stitchTotalMs - accountedMs, stitchTotalMs
        ))
        return lines.joined(separator: "\n")
    }
}

/// Stitch timing + RAM/disk load stats.
struct ChunkStitchResult: Sendable {
    let prefillSeconds: Double
    let ramHits: Int
    let diskLoads: Int
    /// Populated after disk loads so LlamaState can warm the RAM cache.
    let diskSnapshots: [(chunkId: String, data: Data, nTokens: Int)]
    let timing: StitchTimingBreakdown
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

private enum KvMergeKind {
    case passageBody
    case listLabel
}

func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

    batch.n_tokens += 1
}

actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    var is_done: Bool = false

    private var temporary_invalid_cchars: [CChar]

    /// Max generated answer tokens (prefill length is separate).
    var max_gen_tokens: Int32 = 128
    var n_cur: Int32 = 0
    var n_decode: Int32 = 0

    private var inference_start_ns: UInt64 = 0
    private var prefill_end_ns: UInt64 = 0
    private var ttft_ns: UInt64 = 0
    private var first_token_recorded = false
    private var last_metrics: InferenceMetrics?
    private let configured_n_ctx: UInt32

    private func stitchNowNs() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds
    }

    private func stitchElapsedMs(since start: UInt64) -> Double {
        Double(stitchNowNs() - start) / 1_000_000.0
    }

    private func stitchFence(into bucket: inout Double) {
        let t0 = stitchNowNs()
        llama_synchronize(context)
        bucket += stitchElapsedMs(since: t0)
    }

    init(model: OpaquePointer, context: OpaquePointer, nCtx: UInt32) {
        self.model = model
        self.context = context
        self.configured_n_ctx = nCtx
        self.tokens_list = []
        self.batch = llama_batch_init(2048, 0, 1)
        self.temporary_invalid_cchars = []
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.0))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        vocab = llama_model_get_vocab(model)
    }

    deinit {
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_model_free(model)
        llama_free(context)
        llama_backend_free()
    }

    static func create_context(path: String, nCtx: UInt32 = 2048) throws -> LlamaContext {
        llama_backend_init()
        var model_params = llama_model_default_params()
        model_params.use_mmap = true

#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#else
        model_params.n_gpu_layers = 99
#endif

        let model = llama_model_load_from_file(path, model_params)
        guard let model else {
            print("Could not load model at \(path)")
            throw LlamaError.couldNotInitializeContext
        }

        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        let nBatch = Self.recommendedNBatch(for: nCtx)
        print("Using \(n_threads) threads, n_ctx=\(nCtx), n_batch=\(nBatch)")

        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = nCtx
        ctx_params.n_seq_max = 2
        // Fuse SUBSET pass submits all |imp| tokens in one batch (~recomp_ratio × N).
        // WikiMQA 10-passage @ 8192 → ~1136 imp tokens; default 512 is too small.
        ctx_params.n_batch = nBatch
        ctx_params.n_ubatch = nBatch
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED
        // One KV stream: seq_cp(1→0) merges cell metadata instead of wiping stream 0 (required for stitch).
        ctx_params.kv_unified = true
        ctx_params.n_threads       = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)

        let context = llama_init_from_model(model, ctx_params)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }

        return LlamaContext(model: model, context: context, nCtx: nCtx)
    }

    /// Minimum batch for CacheBlend fuse SUBSET pass (|imp| ≈ recomp_ratio × prompt tokens).
    private static func recommendedNBatch(for nCtx: UInt32) -> UInt32 {
        if nCtx <= 2048 { return 512 }
        // 18% of 8192 ≈ 1475; round up to 2048 for WikiMQA 10-passage headroom.
        return min(nCtx, 2048)
    }

    func nCtx() -> UInt32 {
        configured_n_ctx
    }

    /// Live llama context size (use for n_ctx match checks after reload).
    func liveNCtx() -> UInt32 {
        UInt32(llama_n_ctx(context))
    }

    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }

        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }

        return SwiftString
    }

    func get_last_metrics() -> InferenceMetrics? {
        return last_metrics
    }

    /// Decode prompt tokens in n_batch-sized chunks; request logits on the final token only.
    @discardableResult
    private func decodePromptTokens(
        _ tokens: [llama_token],
        pos0: Int32,
        logitsOnLast: Bool
    ) -> Bool {
        guard !tokens.isEmpty else { return true }

        let maxBatch = max(1, Int(llama_n_batch(context)))
        var off = 0
        while off < tokens.count {
            let chunk = min(tokens.count - off, maxBatch)
            llama_batch_clear(&batch)
            for i in 0..<chunk {
                let globalIdx = off + i
                let isLast = logitsOnLast && (globalIdx == tokens.count - 1)
                llama_batch_add(
                    &batch,
                    tokens[globalIdx],
                    pos0 + Int32(globalIdx),
                    [0],
                    isLast
                )
            }
            if llama_decode(context, batch) != 0 {
                print("llama_decode() failed during prompt chunk off=\(off) chunk=\(chunk)")
                return false
            }
            off += chunk
        }
        return true
    }

    /// Ensure logits exist at the last prompt position before sampling.
    private func primeSamplingLogits() -> Bool {
        guard n_cur > 0 else { return false }
        // After decode steps, n_cur advances past tokens_list; re-priming with the
        // prompt tail at the wrong position corrupts KV on Metal.
        guard n_cur <= Int32(tokens_list.count), !tokens_list.isEmpty else {
            return false
        }
        let pos = n_cur - 1
        let token = tokens_list[Int(pos)]
        llama_batch_clear(&batch)
        llama_batch_add(&batch, token, pos, [0], true)
        if llama_decode(context, batch) != 0 {
            print("llama_decode() failed while priming sampling logits")
            return false
        }
        llama_synchronize(context)
        n_cur = pos + 1
        return llama_get_logits_ith(context, -1) != nil
    }

    private func samplingLogitsReady() -> Bool {
        llama_get_logits_ith(context, -1) != nil
    }

    /// Full prefill of `text`, then prepare for decode. Returns prefill duration.
    @discardableResult
    func completion_init(text: String, mode: String = "full_prefill") throws -> Double {
        is_done = false
        n_decode = 0
        first_token_recorded = false
        last_metrics = nil
        inference_start_ns = DispatchTime.now().uptimeNanoseconds

        llama_memory_clear(llama_get_memory(context), true)
        llama_synchronize(context)

        tokens_list = tokenize(text: text, add_bos: true)
        temporary_invalid_cchars = []

        let n_ctx = llama_n_ctx(context)
        let prompt_len = tokens_list.count
        let n_kv_req = prompt_len + Int(max_gen_tokens)

        print("RAG prefill: prompt_tokens=\(prompt_len), n_ctx=\(n_ctx), kv_required=\(n_kv_req)")

        if n_kv_req > Int(n_ctx) {
            print("error: prompt + generation exceeds n_ctx (\(n_ctx))")
            is_done = true
            throw LlamaError.promptTooLong(prompt: prompt_len, maxGen: Int(max_gen_tokens), nCtx: Int(n_ctx))
        }

        if !decodePromptTokens(tokens_list, pos0: 0, logitsOnLast: true) {
            print("llama_decode() failed during prefill")
            is_done = true
            throw LlamaError.samplingLogitsUnavailable
        }
        llama_synchronize(context)

        if !samplingLogitsReady() {
            print("error: prefill finished without logits on last token")
            is_done = true
            throw LlamaError.samplingLogitsUnavailable
        }

        prefill_end_ns = DispatchTime.now().uptimeNanoseconds
        n_cur = Int32(tokens_list.count)

        let prefill_s = Double(prefill_end_ns - inference_start_ns) / 1_000_000_000.0
        print(String(format: "Prefill done: %.3f s (%d tokens)", prefill_s, prompt_len))

        _ = mode // stored in final metrics after decode completes
        return prefill_s
    }

    func completion_loop() -> String {
        if !samplingLogitsReady() {
            if !primeSamplingLogits() {
                print("error: logits unavailable for sampling")
                is_done = true
                return ""
            }
        }

        var new_token_id: llama_token = 0

        new_token_id = llama_sampler_sample(sampling, context, -1)

        if !first_token_recorded {
            ttft_ns = DispatchTime.now().uptimeNanoseconds
            first_token_recorded = true
        }

        let max_pos = Int32(tokens_list.count) + max_gen_tokens
        if llama_vocab_is_eog(vocab, new_token_id) || n_cur >= max_pos {
            is_done = true
            let new_token_str = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            return new_token_str
        }

        let new_token_cchars = token_to_piece(token: new_token_id)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        let new_token_str: String
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else if (0 ..< temporary_invalid_cchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(temporary_invalid_cchars.suffix($0)) + [0]) != nil}) {
            let string = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else {
            new_token_str = ""
        }

        llama_batch_clear(&batch)
        llama_batch_add(&batch, new_token_id, n_cur, [0], true)

        n_decode += 1
        n_cur    += 1

        if llama_decode(context, batch) != 0 {
            print("llama_decode() failed during generation")
            is_done = true
            llama_synchronize(context)
        }

        return new_token_str
    }

    func finalize_metrics(mode: String, decode_start_ns: UInt64) -> InferenceMetrics {
        let decode_end_ns = DispatchTime.now().uptimeNanoseconds
        let prefill_ms = Double(prefill_end_ns - inference_start_ns) / 1_000_000.0
        let ttft_ms = Double((ttft_ns > 0 ? ttft_ns : prefill_end_ns) - inference_start_ns) / 1_000_000.0
        let decode_ms = Double(decode_end_ns - decode_start_ns) / 1_000_000.0
        let decode_count = Int(n_decode)
        let decode_tps = decode_ms > 0 ? Double(decode_count) / (decode_ms / 1000.0) : 0

        let metrics = InferenceMetrics(
            mode: mode,
            promptTokens: tokens_list.count,
            prefillMs: prefill_ms,
            ttftMs: ttft_ms,
            decodeTokens: decode_count,
            decodeMs: decode_ms,
            decodeTps: decode_tps
        )
        last_metrics = metrics
        return metrics
    }

    func bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) -> String {
        var pp_avg: Double = 0
        var tg_avg: Double = 0

        var pp_std: Double = 0
        var tg_std: Double = 0

        for _ in 0..<nr {
            llama_batch_clear(&batch)

            let n_tokens = pp

            for i in 0..<n_tokens {
                llama_batch_add(&batch, 0, Int32(i), [0], false)
            }
            batch.logits[Int(batch.n_tokens) - 1] = 1

            llama_memory_clear(llama_get_memory(context), false)

            let t_pp_start = DispatchTime.now().uptimeNanoseconds / 1000

            if llama_decode(context, batch) != 0 {
                print("llama_decode() failed during prompt")
            }
            llama_synchronize(context)

            let t_pp_end = DispatchTime.now().uptimeNanoseconds / 1000

            llama_memory_clear(llama_get_memory(context), false)

            let t_tg_start = DispatchTime.now().uptimeNanoseconds / 1000

            for i in 0..<tg {
                llama_batch_clear(&batch)

                for j in 0..<pl {
                    llama_batch_add(&batch, 0, Int32(i), [Int32(j)], true)
                }

                if llama_decode(context, batch) != 0 {
                    print("llama_decode() failed during text generation")
                }
                llama_synchronize(context)
            }

            let t_tg_end = DispatchTime.now().uptimeNanoseconds / 1000

            llama_memory_clear(llama_get_memory(context), false)

            let t_pp = Double(t_pp_end - t_pp_start) / 1000000.0
            let t_tg = Double(t_tg_end - t_tg_start) / 1000000.0

            let speed_pp = Double(pp)    / t_pp
            let speed_tg = Double(pl*tg) / t_tg

            pp_avg += speed_pp
            tg_avg += speed_tg

            pp_std += speed_pp * speed_pp
            tg_std += speed_tg * speed_tg

            print("pp \(speed_pp) t/s, tg \(speed_tg) t/s")
        }

        pp_avg /= Double(nr)
        tg_avg /= Double(nr)

        if nr > 1 {
            pp_std = sqrt(pp_std / Double(nr - 1) - pp_avg * pp_avg * Double(nr) / Double(nr - 1))
            tg_std = sqrt(tg_std / Double(nr - 1) - tg_avg * tg_avg * Double(nr) / Double(nr - 1))
        } else {
            pp_std = 0
            tg_std = 0
        }

        let model_desc     = model_info()
        let model_size     = String(format: "%.2f GiB", Double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0)
        let model_n_params = String(format: "%.2f B", Double(llama_model_n_params(model)) / 1e9)
        let backend        = "Metal"
        let pp_avg_str     = String(format: "%.2f", pp_avg)
        let tg_avg_str     = String(format: "%.2f", tg_avg)
        let pp_std_str     = String(format: "%.2f", pp_std)
        let tg_std_str     = String(format: "%.2f", tg_std)

        var result = ""

        result += String("| model | size | params | backend | test | t/s |\n")
        result += String("| --- | --- | --- | --- | --- | --- |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | pp \(pp) | \(pp_avg_str) ± \(pp_std_str) |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | tg \(tg) | \(tg_avg_str) ± \(tg_std_str) |\n")

        return result
    }

    func clear() {
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        is_done = false
        n_decode = 0
        n_cur = 0
        first_token_recorded = false
        last_metrics = nil
        inference_start_ns = 0
        prefill_end_ns = 0
        ttft_ns = 0
        llama_batch_clear(&batch)
        if let mem = llama_get_memory(context) {
            llama_memory_clear(mem, true)
            _ = llama_memory_seq_rm(mem, 1, -1, -1)
        }
        llama_synchronize(context)
    }

    func setMaxGenTokens(_ n: Int32) {
        max_gen_tokens = n
    }

    func promptTokenCount() -> Int {
        tokens_list.count
    }

    // MARK: - Phase B: chunk KV collect / load

    /// Tokenize without adding BOS (for isolated passage chunks).
    func tokenizeChunk(_ text: String) -> [llama_token] {
        tokenize(text: text, add_bos: false)
    }

    /// BOS + system prefix (matches stitch / prefix cache collect).
    func tokenizePrefix(_ prefixText: String) -> [llama_token] {
        tokenize(text: prefixText, add_bos: true)
    }

    /// Prefill `tokens` into a clean KV cache at sequence 0.
    @discardableResult
    func prefillTokens(_ tokens: [llama_token]) throws -> Int {
        guard !tokens.isEmpty else { return 0 }

        llama_memory_clear(llama_get_memory(context), true)
        tokens_list = tokens
        llama_batch_clear(&batch)

        if !decodePromptTokens(tokens, pos0: 0, logitsOnLast: true) {
            throw LlamaError.chunkPrefillFailed(tokenCount: tokens.count)
        }
        llama_synchronize(context)
        n_cur = Int32(tokens.count)
        return tokens.count
    }

    /// Save current KV for `seqId` to a `.bin` file (requires matching `tokens`).
    func saveSequenceState(path: String, tokens: [llama_token], seqId: llama_seq_id = 0) -> Bool {
        let cPath = strdup(path)!
        defer { free(cPath) }

        return tokens.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return false }
            let written = llama_state_seq_save_file(
                context,
                cPath,
                seqId,
                base,
                buffer.count
            )
            return written > 0
        }
    }

    /// Snapshot seq KV state into memory (`llama_state_seq_get_data`).
    func captureSequenceState(seqId: llama_seq_id = 0) -> Data? {
        let size = llama_state_seq_get_size(context, seqId)
        guard size > 0 else { return nil }

        var data = Data(count: Int(size))
        let written: size_t = data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return 0
            }
            return llama_state_seq_get_data(context, base, size, seqId)
        }
        guard written == size else { return nil }
        return data
    }

    /// Restore seq KV from memory snapshot (`llama_state_seq_set_data`).
    func restoreSequenceState(_ data: Data, seqId: llama_seq_id = 0) -> Bool {
        guard !data.isEmpty else { return false }
        let rc: size_t = data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return 0
            }
            return llama_state_seq_set_data(context, base, data.count, seqId)
        }
        return rc > 0
    }

    /// Load KV from `.bin` into `seqId`. Returns token count restored, or nil on failure.
    func loadSequenceState(path: String, seqId: llama_seq_id = 0) -> Int? {
        guard FileManager.default.fileExists(atPath: path) else { return nil }

        let capacity = 4096
        let out = UnsafeMutablePointer<llama_token>.allocate(capacity: capacity)
        defer { out.deallocate() }

        var count: size_t = 0
        let cPath = strdup(path)!
        defer { free(cPath) }

        let read = llama_state_seq_load_file(context, cPath, seqId, out, capacity, &count)
        guard read > 0 else { return nil }
        return Int(count)
    }

    /// Prefill passage text, save KV to `binPath`, return timing stats.
    func collectAndSaveChunk(prefillText: String, binPath: String) throws -> (nTokens: Int, prefillMs: Double, saveMs: Double) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let tokens = tokenizeChunk(prefillText)
        _ = try prefillTokens(tokens)
        let t1 = DispatchTime.now().uptimeNanoseconds

        let saveStart = DispatchTime.now().uptimeNanoseconds
        guard saveSequenceState(path: binPath, tokens: tokens, seqId: 0) else {
            throw LlamaError.chunkSaveFailed(path: binPath)
        }
        let t2 = DispatchTime.now().uptimeNanoseconds

        let prefillMs = Double(t1 - t0) / 1_000_000.0
        let saveMs = Double(t2 - saveStart) / 1_000_000.0
        return (tokens.count, prefillMs, saveMs)
    }

    /// Prefill BOS + system prefix, save KV to `binPath` (matches stitch tokenization).
    func collectAndSavePrefix(prefixText: String, binPath: String) throws -> (nTokens: Int, prefillMs: Double, saveMs: Double) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let tokens = tokenize(text: prefixText, add_bos: true)
        _ = try prefillTokens(tokens)
        let t1 = DispatchTime.now().uptimeNanoseconds

        let saveStart = DispatchTime.now().uptimeNanoseconds
        guard saveSequenceState(path: binPath, tokens: tokens, seqId: 0) else {
            throw LlamaError.chunkSaveFailed(path: binPath)
        }
        let t2 = DispatchTime.now().uptimeNanoseconds

        let prefillMs = Double(t1 - t0) / 1_000_000.0
        let saveMs = Double(t2 - saveStart) / 1_000_000.0
        return (tokens.count, prefillMs, saveMs)
    }

    /// Prefill a list label (`"[n] "`) in isolation and capture KV for RAM cache.
    func collectLabelKvSnapshot(labelText: String) throws -> (data: Data, nTokens: Int) {
        let tokens = tokenize(text: labelText, add_bos: false)
        guard !tokens.isEmpty else {
            throw LlamaError.chunkPrefillFailed(tokenCount: 0)
        }
        _ = try prefillTokens(tokens)
        guard let data = captureSequenceState(seqId: 0) else {
            throw LlamaError.couldNotLoadCachedChunk
        }
        return (data, tokens.count)
    }

    /// Restore a cached KV blob (prefix or chunk body) into stream 0 at `position`.
    private func mergeCachedKVBlob(
        mem: OpaquePointer,
        itemRam: Data?,
        itemBinPath: String,
        nTokens: Int,
        position: Int32,
        chunkId: String,
        ramHits: inout Int,
        diskLoads: inout Int,
        diskSnapshots: inout [(chunkId: String, data: Data, nTokens: Int)],
        timing: inout StitchTimingBreakdown,
        mergeKind: KvMergeKind = .passageBody
    ) throws -> Int {
        timing.mergeCount += 1
        let loadedTokens: Int

        if let ram = itemRam {
            let t0 = stitchNowNs()
            let ok = restoreSequenceState(ram, seqId: 1)
            let deserMs = stitchElapsedMs(since: t0)
            if mergeKind == .listLabel {
                timing.labelCacheMs += deserMs
                stitchFence(into: &timing.labelCacheMs)
            } else {
                timing.deserializeRamMs += deserMs
                stitchFence(into: &timing.deserializeRamMs)
            }
            guard ok else { throw LlamaError.couldNotLoadCachedChunk }
            loadedTokens = nTokens
            ramHits += 1
            if mergeKind != .listLabel {
                timing.ramHits += 1
            }
        } else {
            let t0 = stitchNowNs()
            guard let fromDisk = loadSequenceState(path: itemBinPath, seqId: 1) else {
                throw LlamaError.couldNotLoadCachedChunk
            }
            let deserMs = stitchElapsedMs(since: t0)
            if mergeKind == .listLabel {
                timing.labelCacheMs += deserMs
                stitchFence(into: &timing.labelCacheMs)
            } else {
                timing.deserializeDiskMs += deserMs
                stitchFence(into: &timing.deserializeDiskMs)
            }
            loadedTokens = fromDisk
            diskLoads += 1
            timing.diskLoads += 1
            let snapStart = stitchNowNs()
            if let snap = captureSequenceState(seqId: 1) {
                diskSnapshots.append((chunkId, snap, loadedTokens))
            }
            timing.snapshotWarmMs += stitchElapsedMs(since: snapStart)
        }

        let mergeStart = stitchNowNs()
        llama_memory_seq_add(mem, 1, 0, -1, position)
        llama_memory_seq_cp(mem, 1, 0, 0, -1)
        _ = llama_memory_seq_rm(mem, 1, -1, -1)
        let memcpyBlockMs = stitchElapsedMs(since: mergeStart)
        if mergeKind == .listLabel {
            timing.labelCacheMs += memcpyBlockMs
            stitchFence(into: &timing.labelCacheMs)
        } else {
            timing.memcpyMs += memcpyBlockMs
            stitchFence(into: &timing.memcpyMs)
        }
        return loadedTokens
    }

    /// Phase C: stitch cached prefix/chunks and prefill labels + question.
    /// Uses RAM snapshot when `ramState` is set; otherwise loads `.bin`.
    @discardableResult
    func completion_init_with_cached_chunks(
        prefixText: String,
        prefixItem: PrefixStitchItem? = nil,
        chunkItems: [ChunkStitchItem],
        questionSuffix: String,
        mode: String = "phase_c_reuse",
        prefillQuestion: Bool = true
    ) throws -> ChunkStitchResult {
        is_done = false
        n_decode = 0
        first_token_recorded = false
        last_metrics = nil
        inference_start_ns = DispatchTime.now().uptimeNanoseconds
        temporary_invalid_cchars = []

        guard let mem = llama_get_memory(context) else {
            throw LlamaError.couldNotInitializeContext
        }

        var timing = StitchTimingBreakdown()
        let clearStart = stitchNowNs()
        llama_memory_clear(mem, true)
        timing.clearMs += stitchElapsedMs(since: clearStart)
        stitchFence(into: &timing.clearMs)

        var position: Int32 = 0
        var ramHits = 0
        var diskLoads = 0
        var diskSnapshots: [(chunkId: String, data: Data, nTokens: Int)] = []

        // 1) System prefix: cached KV or fresh GPU prefill.
        if let prefixItem {
            let loaded = try mergeCachedKVBlob(
                mem: mem,
                itemRam: prefixItem.ramState,
                itemBinPath: prefixItem.binPath,
                nTokens: prefixItem.nTokens,
                position: position,
                chunkId: prefixItem.chunkId,
                ramHits: &ramHits,
                diskLoads: &diskLoads,
                diskSnapshots: &diskSnapshots,
                timing: &timing
            )
            position += Int32(loaded)
        } else {
            let prefixTokens = tokenize(text: prefixText, add_bos: true)
            if !prefixTokens.isEmpty {
                let gpuStart = stitchNowNs()
                if !decodePromptTokens(prefixTokens, pos0: position, logitsOnLast: false) {
                    throw LlamaError.couldNotInitializeContext
                }
                timing.prefixGpuMs += stitchElapsedMs(since: gpuStart)
                timing.prefixTokens += prefixTokens.count
                stitchFence(into: &timing.prefixGpuMs)
                position += Int32(prefixTokens.count)
            }
        }

        // 2) For each chunk: merge cached `[n] ` label KV or GPU prefill, then stitch body KV.
        for item in chunkItems {
            if let labelRam = item.labelRamState, item.labelNTokens > 0 {
                let loaded = try mergeCachedKVBlob(
                    mem: mem,
                    itemRam: labelRam,
                    itemBinPath: "",
                    nTokens: item.labelNTokens,
                    position: position,
                    chunkId: "label-\(item.chunkId)",
                    ramHits: &ramHits,
                    diskLoads: &diskLoads,
                    diskSnapshots: &diskSnapshots,
                    timing: &timing,
                    mergeKind: .listLabel
                )
                timing.labelTokens += loaded
                position += Int32(loaded)
            } else if !item.labelText.isEmpty {
                let labelTokens = tokenize(text: item.labelText, add_bos: false)
                if !labelTokens.isEmpty {
                    let gpuStart = stitchNowNs()
                    if !decodePromptTokens(labelTokens, pos0: position, logitsOnLast: false) {
                        throw LlamaError.couldNotInitializeContext
                    }
                    timing.labelGpuMs += stitchElapsedMs(since: gpuStart)
                    timing.labelTokens += labelTokens.count
                    stitchFence(into: &timing.labelGpuMs)
                    position += Int32(labelTokens.count)
                }
            }

            let loadedTokens = try mergeCachedKVBlob(
                mem: mem,
                itemRam: item.ramState,
                itemBinPath: item.binPath,
                nTokens: item.nTokens,
                position: position,
                chunkId: item.chunkId,
                ramHits: &ramHits,
                diskLoads: &diskLoads,
                diskSnapshots: &diskSnapshots,
                timing: &timing
            )
            position += Int32(loadedTokens)
        }

        // 3) Prefill question suffix at tail (skipped when fuse will prefill question later).
        if prefillQuestion {
            let questionTokens = tokenize(text: questionSuffix, add_bos: false)
            guard !questionTokens.isEmpty else {
                throw LlamaError.couldNotInitializeContext
            }
            let questionStart = stitchNowNs()
            if !decodePromptTokens(questionTokens, pos0: position, logitsOnLast: true) {
                throw LlamaError.couldNotInitializeContext
            }
            timing.questionGpuMs += stitchElapsedMs(since: questionStart)
            timing.questionTokens += questionTokens.count
            stitchFence(into: &timing.questionGpuMs)

            n_cur = position + Int32(questionTokens.count)
            if !samplingLogitsReady() {
                throw LlamaError.samplingLogitsUnavailable
            }
            prefill_end_ns = DispatchTime.now().uptimeNanoseconds

            let promptTokenCount = Int(position) + questionTokens.count
            let prefixNote = prefixItem != nil ? "cached-prefix" : "fresh-prefix"
            let prefill_s = Double(prefill_end_ns - inference_start_ns) / 1_000_000_000.0
            print(String(
                format: "Prefill done: %.3f s (%d tokens) [%@] %@ stitch RAM %d disk %d",
                prefill_s, promptTokenCount, mode, prefixNote, ramHits, diskLoads
            ))
        } else {
            n_cur = position
            prefill_end_ns = DispatchTime.now().uptimeNanoseconds
            let prefill_s = Double(prefill_end_ns - inference_start_ns) / 1_000_000_000.0
            print(String(
                format: "Stitch done: %.3f s (%d tokens, question deferred) [%@] RAM %d disk %d",
                prefill_s, position, mode, ramHits, diskLoads
            ))
        }

        let prefill_s = Double(prefill_end_ns - inference_start_ns) / 1_000_000_000.0
        return ChunkStitchResult(
            prefillSeconds: prefill_s,
            ramHits: ramHits,
            diskLoads: diskLoads,
            diskSnapshots: diskSnapshots,
            timing: timing
        )
    }

    /// Legacy stitch API (disk-only). Prefer `chunkItems` + RAM snapshots.
    @discardableResult
    func completion_init_with_cached_chunks(
        prefixText: String,
        chunkBinPathsInOrder: [String],
        questionSuffix: String,
        mode: String = "phase_c_reuse"
    ) throws -> Double {
        let items = chunkBinPathsInOrder.enumerated().map { idx, path in
            ChunkStitchItem(
                chunkId: "disk-\(idx)",
                labelText: "",
                binPath: path,
                nTokens: 0,
                ramState: nil,
                labelRamState: nil,
                labelNTokens: 0
            )
        }
        let result = try completion_init_with_cached_chunks(
            prefixText: prefixText,
            prefixItem: nil,
            chunkItems: items,
            questionSuffix: questionSuffix,
            mode: mode
        )
        return result.prefillSeconds
    }

    /// Set prompt token ids after stitch (must match KV positions). Call before fuse/decode.
    func setStitchTokenList(_ tokens: [llama_token]) {
        tokens_list = tokens
        n_cur = Int32(tokens.count)
    }

    /// GPU prefill question suffix at current KV tail (after stitch/fuse). Returns wall ms.
    @discardableResult
    func prefillQuestionSuffix(_ questionSuffix: String) throws -> Double {
        let questionTokens = tokenize(text: questionSuffix, add_bos: false)
        guard !questionTokens.isEmpty else {
            throw LlamaError.couldNotInitializeContext
        }
        let pos = n_cur
        let t0 = DispatchTime.now().uptimeNanoseconds
        if !decodePromptTokens(questionTokens, pos0: pos, logitsOnLast: true) {
            throw LlamaError.couldNotInitializeContext
        }
        llama_synchronize(context)
        tokens_list.append(contentsOf: questionTokens)
        n_cur = pos + Int32(questionTokens.count)
        prefill_end_ns = DispatchTime.now().uptimeNanoseconds
        if !samplingLogitsReady() {
            throw LlamaError.samplingLogitsUnavailable
        }
        return Double(prefill_end_ns - t0) / 1_000_000.0
    }

    // MARK: - Phase E: CacheBlend fuse

    /// Run `llama_cache_blend_fuse` on stitched KV. HKVD runs inside fuse when `impIndices` is nil.
    func cacheBlendFuse(
        tokens: [llama_token],
        suffixLen: UInt32,
        checkLayer: Int32? = nil,
        recompRatio: Float = 0.18,
        mode: CacheBlendFuseMode = .graph,
        impIndices: [UInt32]? = nil,
        requireSamplingLogits: Bool = true
    ) throws -> CacheBlendFuseResult {
        let layer = checkLayer ?? defaultHkvdCheckLayer()
        tokens_list = tokens

        let nCtx = llama_n_ctx(context)
        if tokens.count + Int(max_gen_tokens) > Int(nCtx) {
            throw LlamaError.promptTooLong(
                prompt: tokens.count,
                maxGen: Int(max_gen_tokens),
                nCtx: Int(nCtx)
            )
        }

        var params = llama_cache_blend_default_params()
        params.mode = mode.cMode
        params.check_layer = layer
        params.recomp_ratio = recompRatio
        params.suffix_len = suffixLen
        params.n_tokens = UInt32(tokens.count)
        params.seq_id = 0

        var impOut = [UInt32](repeating: 0, count: tokens.count)
        var nImpOut: UInt32 = 0

        let t0 = DispatchTime.now().uptimeNanoseconds
        let nBatch = llama_n_batch(context)
        print(String(format: "cache_blend_fuse: n_tokens=%d n_batch=%u mode=%@ layer=%d",
                     tokens.count, nBatch, mode.label, layer))
        let rc: Int32 = tokens.withUnsafeBufferPointer { tokPtr in
            params.tokens = tokPtr.baseAddress

            if let imp = impIndices, !imp.isEmpty {
                return imp.withUnsafeBufferPointer { impPtr in
                    params.imp_indices = impPtr.baseAddress
                    params.n_imp = UInt32(imp.count)
                    params.imp_indices_out = nil
                    params.n_imp_out = nil
                    return llama_cache_blend_fuse(context, &params)
                }
            }

            params.imp_indices = nil
            params.n_imp = 0
            return impOut.withUnsafeMutableBufferPointer { outPtr in
                params.imp_indices_out = outPtr.baseAddress
                params.imp_indices_out_capacity = UInt32(outPtr.count)
                return withUnsafeMutablePointer(to: &nImpOut) { nImpPtr in
                    params.n_imp_out = nImpPtr
                    return llama_cache_blend_fuse(context, &params)
                }
            }
        }
        guard rc == 0 else {
            print("llama_cache_blend_fuse failed: rc=\(rc)")
            throw LlamaError.cacheBlendFuseFailed(rc)
        }

        let indices: [UInt32]
        if let imp = impIndices, !imp.isEmpty {
            indices = imp
        } else {
            indices = Array(impOut.prefix(Int(nImpOut)))
        }

        n_cur = Int32(tokens.count)
        llama_synchronize(context)

        if requireSamplingLogits {
            if !samplingLogitsReady() {
                if !primeSamplingLogits() {
                    print("cache_blend_fuse: failed to prime sampling logits")
                    throw LlamaError.samplingLogitsUnavailable
                }
            }
        }

        let fuseMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000.0
        prefill_end_ns = DispatchTime.now().uptimeNanoseconds

        return CacheBlendFuseResult(
            mode: mode,
            checkLayer: layer,
            nTokens: UInt32(tokens.count),
            suffixLen: suffixLen,
            recompRatio: recompRatio,
            indices: indices,
            fuseMs: fuseMs
        )
    }

    // MARK: - Phase D: layer KV export + HKVD

    /// Token sequence produced by the Phase C stitch path (must match stitched KV positions).
    func buildStitchTokenList(
        prefixText: String,
        passages: [String],
        questionSuffix: String,
        includeQuestion: Bool = true
    ) -> [llama_token] {
        var tokens = tokenize(text: prefixText, add_bos: true)
        for (idx, passage) in passages.enumerated() {
            let label = "[\(idx + 1)] "
            tokens.append(contentsOf: tokenize(text: label, add_bos: false))
            // Match ChunkStore.prefillText / collectAndSaveChunk tokenization (body only).
            let prefillText = passage.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
            tokens.append(contentsOf: tokenizeChunk(prefillText))
        }
        if includeQuestion {
            tokens.append(contentsOf: tokenize(text: questionSuffix, add_bos: false))
        }
        return tokens
    }

    /// HKVD check layer: layer index 1 (second block; 0 = first).
    func defaultHkvdCheckLayer() -> Int32 {
        return 1
    }

    /// Export one layer's V cache as float32 `[n_tokens][n_kv_heads][head_dim]`.
    /// Pass `nPos` = prompt length to export positions `[0, nPos)` (required after stitch).
    func exportLayerVFloat(seqId: llama_seq_id = 0, layerIdx: Int32, nPos: Int32 = 0) throws -> LayerVExport {
        var nFloats: size_t = 0
        var nTokens: UInt32 = 0
        var nKvHeads: UInt32 = 0
        var headDim: UInt32 = 0

        let sizeRc = llama_state_seq_get_layer_v_f32(
            context,
            seqId,
            layerIdx,
            nil,
            0,
            &nFloats,
            &nTokens,
            &nKvHeads,
            &headDim,
            nPos
        )
        guard sizeRc == 0, nFloats > 0 else {
            throw LlamaError.hkvdExportFailed(sizeRc)
        }

        let buffer = UnsafeMutablePointer<Float>.allocate(capacity: nFloats)
        defer { buffer.deallocate() }

        let rc = llama_state_seq_get_layer_v_f32(
            context,
            seqId,
            layerIdx,
            buffer,
            nFloats * MemoryLayout<Float>.size,
            &nFloats,
            &nTokens,
            &nKvHeads,
            &headDim,
            nPos
        )
        guard rc == 0 else {
            throw LlamaError.hkvdExportFailed(rc)
        }

        let values = Array(UnsafeBufferPointer(start: buffer, count: nFloats))
        return LayerVExport(
            values: values,
            nTokens: nTokens,
            nKvHeads: nKvHeads,
            headDim: headDim
        )
    }

    /// CacheBlend-style HKVD index selection from two f32 V tensors.
    func computeHkvdIndices(
        vOld: [Float],
        vNew: [Float],
        nTokens: UInt32,
        nKvHeads: UInt32,
        headDim: UInt32,
        suffixLen: UInt32,
        recompRatio: Float
    ) throws -> [UInt32] {
        let stride = Int(nKvHeads) * Int(headDim)
        let expected = Int(nTokens) * stride
        guard vOld.count == expected, vNew.count == expected else {
            throw LlamaError.hkvdShapeMismatch(
                oldTokenCount: vOld.count / max(stride, 1),
                newTokenCount: vNew.count / max(stride, 1)
            )
        }

        let prefixLen = Int(nTokens) - Int(suffixLen)
        let topk = Int(Float(prefixLen) * recompRatio)
        let capacity = topk + Int(suffixLen)
        var out = [UInt32](repeating: 0, count: capacity)
        var outCount: size_t = 0

        let rc = vOld.withUnsafeBufferPointer { oldPtr in
            vNew.withUnsafeBufferPointer { newPtr in
                out.withUnsafeMutableBufferPointer { outPtr in
                    llama_compute_hkvd_indices(
                        oldPtr.baseAddress,
                        newPtr.baseAddress,
                        nTokens,
                        nKvHeads,
                        headDim,
                        suffixLen,
                        recompRatio,
                        outPtr.baseAddress,
                        outPtr.count,
                        &outCount
                    )
                }
            }
        }
        guard rc == 0 else {
            throw LlamaError.hkvdComputeFailed(rc)
        }

        return Array(out.prefix(Int(outCount)))
    }

    /// Compare stitched KV (current ctx) vs fresh prefill of the same stitch token sequence.
    /// Caller must re-run stitch afterward before decode.
    func probeHkvdIndices(
        checkLayer: Int32,
        prefixText: String,
        passages: [String],
        questionSuffix: String,
        suffixTokenCount: UInt32,
        recompRatio: Float
    ) throws -> HkvdProbeResult {
        let stitchTokens = buildStitchTokenList(
            prefixText: prefixText,
            passages: passages,
            questionSuffix: questionSuffix
        )
        let nPos = Int32(stitchTokens.count)

        let t0 = DispatchTime.now().uptimeNanoseconds
        let vOldExport = try exportLayerVFloat(seqId: 0, layerIdx: checkLayer, nPos: nPos)
        let t1 = DispatchTime.now().uptimeNanoseconds

        llama_memory_clear(llama_get_memory(context), true)
        let prefillStart = DispatchTime.now().uptimeNanoseconds
        _ = try prefillTokens(stitchTokens)
        let t2 = DispatchTime.now().uptimeNanoseconds

        let vNewExport = try exportLayerVFloat(seqId: 0, layerIdx: checkLayer, nPos: nPos)
        let t3 = DispatchTime.now().uptimeNanoseconds

        if vOldExport.nTokens != vNewExport.nTokens {
            throw LlamaError.hkvdShapeMismatch(
                oldTokenCount: Int(vOldExport.nTokens),
                newTokenCount: Int(vNewExport.nTokens)
            )
        }

        let t4 = DispatchTime.now().uptimeNanoseconds
        let indices = try computeHkvdIndices(
            vOld: vOldExport.values,
            vNew: vNewExport.values,
            nTokens: vOldExport.nTokens,
            nKvHeads: vOldExport.nKvHeads,
            headDim: vOldExport.headDim,
            suffixLen: suffixTokenCount,
            recompRatio: recompRatio
        )
        let t5 = DispatchTime.now().uptimeNanoseconds

        return HkvdProbeResult(
            checkLayer: checkLayer,
            nTokens: vOldExport.nTokens,
            nKvHeads: vOldExport.nKvHeads,
            headDim: vOldExport.headDim,
            suffixLen: suffixTokenCount,
            recompRatio: recompRatio,
            indices: indices,
            exportOldMs: Double(t1 - t0) / 1_000_000.0,
            fullPrefillMs: Double(t2 - prefillStart) / 1_000_000.0,
            exportNewMs: Double(t3 - t2) / 1_000_000.0,
            computeMs: Double(t5 - t4) / 1_000_000.0
        )
    }

    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, false)

        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()

        return swiftTokens
    }

    private func token_to_piece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(vocab, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(vocab, token, newResult, -nTokens, 0, false)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
}
