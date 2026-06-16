import Foundation
import llama

enum LlamaError: Error {
    case couldNotInitializeContext
}

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

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
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

    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        self.tokens_list = []
        self.batch = llama_batch_init(512, 0, 1)
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

    static func create_context(path: String) throws -> LlamaContext {
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
        print("Using \(n_threads) threads")

        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = 2048
        ctx_params.n_threads       = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)

        let context = llama_init_from_model(model, ctx_params)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }

        return LlamaContext(model: model, context: context)
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

    /// Full prefill of `text`, then prepare for decode. Returns prefill duration.
    @discardableResult
    func completion_init(text: String, mode: String = "full_prefill") -> Double {
        is_done = false
        n_decode = 0
        first_token_recorded = false
        last_metrics = nil
        inference_start_ns = DispatchTime.now().uptimeNanoseconds

        tokens_list = tokenize(text: text, add_bos: true)
        temporary_invalid_cchars = []

        let n_ctx = llama_n_ctx(context)
        let prompt_len = tokens_list.count
        let n_kv_req = prompt_len + Int(max_gen_tokens)

        print("RAG prefill: prompt_tokens=\(prompt_len), n_ctx=\(n_ctx), kv_required=\(n_kv_req)")

        if n_kv_req > Int(n_ctx) {
            print("warning: prompt + generation exceeds n_ctx (\(n_ctx))")
        }

        llama_batch_clear(&batch)

        for i in 0..<tokens_list.count {
            llama_batch_add(&batch, tokens_list[i], Int32(i), [0], false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1

        if llama_decode(context, batch) != 0 {
            print("llama_decode() failed during prefill")
        }
        llama_synchronize(context)

        prefill_end_ns = DispatchTime.now().uptimeNanoseconds
        n_cur = batch.n_tokens

        let prefill_s = Double(prefill_end_ns - inference_start_ns) / 1_000_000_000.0
        print(String(format: "Prefill done: %.3f s (%d tokens)", prefill_s, prompt_len))

        _ = mode // stored in final metrics after decode completes
        return prefill_s
    }

    func completion_loop() -> String {
        var new_token_id: llama_token = 0

        new_token_id = llama_sampler_sample(sampling, context, batch.n_tokens - 1)

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
        llama_memory_clear(llama_get_memory(context), true)
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
