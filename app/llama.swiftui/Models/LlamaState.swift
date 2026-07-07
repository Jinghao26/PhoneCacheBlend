import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

/// Phase A default model — Qwen2.5-1.5B-Instruct Q4_K_M
enum PhoneCacheBlendConfig {
    static let qwenFilename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    static let qwenDownloadURL =
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

    static let systemPrefix = """
        You are a helpful assistant. Answer the question using only the passages below. Be concise.

        Passages:
        """

    static let questionPrefix = "\n\nQuestion: "
    static let answerPrefix = "\nAnswer:"

    /// Phase D: run HKVD probe after stitch (debug; extra full prefill). Off when fuse is on.
    static let enableHkvdProbe = false
    static let hkvdRecompRatio: Float = 0.18

    /// Phase E: run CacheBlend fuse after stitch before decode.
    static let enableCacheBlendFuse = true
    /// Phase E: selective multi-pass fuse (default). Falls back to TOKEN_RECOMPUTE on failure.
    static let cacheBlendFuseMode: CacheBlendFuseMode = .graph

    /// Phase C+: keep chunk KV in RAM to skip disk reload during stitch.
    static let enableRamHotStitch = true

    /// Cache BOS + system prefix KV on disk/RAM (skip GPU prefill at stitch when HIT).
    static let enableSystemPrefixCache = true

    /// Cache `[n] ` list-label KV in RAM; stitch merges instead of GPU prefill per chunk.
    static let enableLabelKvCache = true

    /// Prefill question suffix after fuse (suffix_len=0 at fuse) instead of during stitch.
    static let enableQuestionPrefillAfterFuse = true

    // MARK: - Chunk KV cache capacity (benchmarking headroom)

    /// Max passage (+ prefix) KV blobs on disk (FIFO). Simple/Harder_test: 36 passages + prefix ≈ 37.
    static let chunkDiskMaxEntries = 128
    /// Max passage chunk KV snapshots in RAM for hot stitch (avoid disk reload during stitch).
    static let chunkRamMaxEntries = 96

    // MARK: - WikiMQA benchmark capacity (set `wikiBlobCeilingPassages` after @8192 ceiling run)

    /// Passages cached at jetsam limit with n_ctx=8192 (device measurement). 0 = not yet measured.
    static var wikiBlobCeilingPassages: Int = 128
    /// Disk FIFO for WikiMQA 200 (~1,055 unique passages + prefix margin).
    static let wikiChunkDiskMaxEntries = 1280

    /// RAM hot slots for WikiMQA benchmarking (~64 × 17 MB ≈ 1.1 GB; leaves room for n_ctx=8192 query).
    static let wikiChunkRamMaxEntries = 64

    /// Default context for Simple_test and normal Ask flow.
    static let nCtxDefault: UInt32 = 2048
    /// Expanded context for Harder_test mega prompts (33 chunks).
    static let nCtxHarder: UInt32 = 4096
    /// WikiMQA full query (~5.7k tok prompt); required for stitch/fuse on 10 passages.
    static let nCtxWikiMQA: UInt32 = 8192

    static func ramStressNCtx(scale: RamStressScale, stitchProbe: Bool = false) -> UInt32 {
        _ = stitchProbe
        switch scale {
        case .simple:
            return nCtxDefault
        case .wikiScale:
            return nCtxWikiMQA
        }
    }

    static func ramStressRampSteps(for scale: RamStressScale) -> [Int] {
        switch scale {
        case .simple:
            return ramStressRampStepsSimple
        case .wikiScale:
            return ramStressRampStepsWiki8192
        }
    }

    /// WikiMQA query sizes for stitch probe @ n_ctx=8192.
    static let ramStressWikiStitchProbePassages = [4, 8, 10]

    /// Default background cache size for WikiMQA query-probe-only.
    static let ramStressWikiQueryProbeCacheDefault = 64
    static let ramStressWikiQueryProbeCacheSizes = [32, 64, 96]

    /// Disk FIFO cap while ramping passage KV blobs (avoid eviction before jetsam).
    static let ramStressDiskMaxEntries = 512
    /// RAM hot-cache cap while ramping (no eviction until this limit).
    static let ramStressRamMaxEntries = 512

    /// Simple_test blob ramp @ n_ctx=2048.
    static let ramStressRampStepsSimple: [Int] = [
        4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96,
        104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256
    ]

    /// WikiMQA blob ramp @ n_ctx=8192 (finer steps — expect lower ceiling than 2048 ramp).
    static let ramStressRampStepsWiki8192: [Int] = [
        4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64,
        68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 128
    ]
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    @Published var isInferring = false
    @Published var chunkCacheSummary = ""

    private var llamaContext: LlamaContext?
    private var loadedModelFilename: String?
    private var loadedModelPath: String?
    private var chunkStore: ChunkStore?
    private let chunkRamCache = ChunkRamCache()
    private let labelRamCache = ChunkRamCache(maxEntries: ChunkRamCacheConfig.maxLabelEntries)

    init() {
        chunkStore = ChunkStore(documentsDirectory: getDocumentsDirectory())
        chunkCacheSummary = chunkStore?.summaryLine() ?? ""
        loadModelsFromDisk()
        loadDefaultModels()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(
                at: documentsURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            )
            for modelURL in modelURLs where modelURL.pathExtension == "gguf" {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                if !downloadedModels.contains(where: { $0.filename == modelURL.lastPathComponent }) {
                    downloadedModels.append(Model(
                        name: modelName,
                        url: "",
                        filename: modelURL.lastPathComponent,
                        status: "downloaded"
                    ))
                }
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        let qwenURL = getDocumentsDirectory().appendingPathComponent(PhoneCacheBlendConfig.qwenFilename)
        if FileManager.default.fileExists(atPath: qwenURL.path) {
            do {
                try loadModel(modelUrl: qwenURL)
            } catch {
                messageLog += "Failed to load Qwen: \(error)\n"
            }
        } else {
            messageLog += "Download Qwen2.5-1.5B Q4_K_M from Models screen.\n"
        }

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if !FileManager.default.fileExists(atPath: fileURL.path) {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                if !undownloadedModels.contains(where: { $0.filename == model.filename }) {
                    undownloadedModels.append(undownloadedModel)
                }
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private let defaultModels: [Model] = [
        Model(
            name: "Qwen2.5-1.5B-Instruct (Q4_K_M, ~1 GiB) ★",
            url: PhoneCacheBlendConfig.qwenDownloadURL,
            filename: PhoneCacheBlendConfig.qwenFilename,
            status: "download"
        ),
    ]

    func loadModel(modelUrl: URL?) throws {
        guard let modelUrl else {
            messageLog += "No model file selected.\n"
            return
        }
        messageLog += "Loading model...\n"
        loadedModelPath = modelUrl.path()
        llamaContext = try LlamaContext.create_context(
            path: modelUrl.path(),
            nCtx: PhoneCacheBlendConfig.nCtxDefault
        )
        loadedModelFilename = modelUrl.lastPathComponent
        chunkRamCache.removeAll()
        labelRamCache.removeAll()
        messageLog += "Loaded: \(modelUrl.lastPathComponent) (n_ctx=\(PhoneCacheBlendConfig.nCtxDefault))\n"
        updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
    }

    /// Reload model when n_ctx must change (e.g. 2048 for Simple vs 4096 for Harder).
    /// Releases the old context **before** allocating the new one to avoid a 2× model/KV peak
    /// (Metal OOM / NULL buffer crash when large chunk caches are already resident).
    private func ensureContextCapacity(nCtx: UInt32) async throws {
        guard loadedModelPath != nil else { return }
        if let llamaContext, await llamaContext.nCtx() == nCtx {
            return
        }
        messageLog += "Reloading model with n_ctx=\(nCtx)…\n"
        chunkRamCache.removeAll()
        labelRamCache.removeAll()
        try await reloadLlamaContextAtNCtx(nCtx)
    }

    /// Drop and recreate the llama context at `nCtx` (releases Metal KV without a 2× peak).
    private func reloadLlamaContextAtNCtx(_ nCtx: UInt32) async throws {
        guard let path = loadedModelPath else { return }
        if let old = llamaContext {
            await old.clear()
        }
        llamaContext = nil
        await Task.yield()
        llamaContext = try LlamaContext.create_context(path: path, nCtx: nCtx)
    }

    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.filename == modelName }
    }

    /// Build full RAG prompt: system + passages + question.
    static func buildRagPrompt(passages: [String], question: String) -> String {
        let trimmedPassages = passages
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let body = trimmedPassages.enumerated().map { index, passage in
            "[\(index + 1)] \(passage)"
        }.joined(separator: "\n\n")

        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        return PhoneCacheBlendConfig.systemPrefix
            + body
            + PhoneCacheBlendConfig.questionPrefix
            + q
            + PhoneCacheBlendConfig.answerPrefix
    }

    /// List index label prefilled fresh at stitch time (not part of cached body KV).
    static func formatChunkLabel(index: Int) -> String {
        "[\(index + 1)] "
    }

    /// Chunk text format used for baseline full prefill only.
    static func formatPassageChunk(index: Int, passage: String) -> String {
        formatChunkLabel(index: index)
            + passage.trimmingCharacters(in: .whitespacesAndNewlines)
            + "\n\n"
    }

    /// Ensure each chunk KV is on disk (save on miss, FIFO eviction).
    private func ensureSystemPrefixCached(llamaContext: LlamaContext) async -> ChunkCacheOpResult? {
        guard PhoneCacheBlendConfig.enableSystemPrefixCache,
              let chunkStore,
              let modelFilename = loadedModelFilename else { return nil }

        let prefixText = PhoneCacheBlendConfig.systemPrefix
        let chunkId = ChunkStore.prefixChunkId(modelFilename: modelFilename, prefixText: prefixText)
        let currentNCtx = await llamaContext.nCtx()

        if chunkStore.hasValidCache(chunkId: chunkId, forNCtx: currentNCtx) {
            try? chunkStore.touchExisting(chunkId: chunkId)
            let nTokens: Int
            if let cached = chunkStore.metadata(for: chunkId)?.nTokens {
                nTokens = cached
            } else {
                nTokens = await llamaContext.tokenizePrefix(prefixText).count
            }
            return ChunkCacheOpResult(
                chunkId: chunkId,
                cacheHit: true,
                nTokens: nTokens,
                prefillMs: nil,
                saveMs: nil,
                evictedChunkIds: []
            )
        }

        let binPath = chunkStore.binURL(chunkId: chunkId).path
        do {
            let stats = try await llamaContext.collectAndSavePrefix(
                prefixText: prefixText,
                binPath: binPath
            )

            let preview = String(prefixText.prefix(80).replacingOccurrences(of: "\n", with: " "))
            let metadata = ChunkMetadata(
                chunkId: chunkId,
                modelFilename: modelFilename,
                nTokens: stats.nTokens,
                prefillTextPreview: preview,
                createdAt: Date(),
                captureNCtx: currentNCtx
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let metaData = try encoder.encode(metadata)
            try metaData.write(to: chunkStore.metaURL(chunkId: chunkId), options: .atomic)

            let evicted = (try? chunkStore.registerSavedChunk(metadata)) ?? []
            chunkRamCache.removeAll(evicted)

            if PhoneCacheBlendConfig.enableRamHotStitch,
               let blob = await llamaContext.captureSequenceState(seqId: 0) {
                chunkRamCache.put(chunkId: chunkId, stateData: blob, nTokens: stats.nTokens)
            }

            return ChunkCacheOpResult(
                chunkId: chunkId,
                cacheHit: false,
                nTokens: stats.nTokens,
                prefillMs: stats.prefillMs,
                saveMs: stats.saveMs,
                evictedChunkIds: evicted
            )
        } catch {
            messageLog += "Prefix cache save failed: \(chunkId.prefix(8))… \(error)\n"
            return nil
        }
    }

    /// Ensure each chunk KV is on disk (save on miss, FIFO eviction).
    private func ensurePassagesCached(
        passages: [String],
        llamaContext: LlamaContext
    ) async -> [ChunkCacheOpResult] {
        guard let chunkStore, let modelFilename = loadedModelFilename else { return [] }

        var results: [ChunkCacheOpResult] = []
        let currentNCtx = await llamaContext.nCtx()

        for passage in passages {
            let chunkId = ChunkStore.chunkId(modelFilename: modelFilename, passageContent: passage)
            let prefillText = ChunkStore.prefillText(for: passage)

            if chunkStore.hasValidCache(chunkId: chunkId, forNCtx: currentNCtx) {
                try? chunkStore.touchExisting(chunkId: chunkId)
                let nTokens: Int
                if let cached = chunkStore.metadata(for: chunkId)?.nTokens {
                    nTokens = cached
                } else {
                    nTokens = await llamaContext.tokenizeChunk(prefillText).count
                }
                results.append(ChunkCacheOpResult(
                    chunkId: chunkId,
                    cacheHit: true,
                    nTokens: nTokens,
                    prefillMs: nil,
                    saveMs: nil,
                    evictedChunkIds: []
                ))
                continue
            }

            let binPath = chunkStore.binURL(chunkId: chunkId).path
            do {
                let stats = try await llamaContext.collectAndSaveChunk(
                    prefillText: prefillText,
                    binPath: binPath
                )

                let trimmed = passage.trimmingCharacters(in: .whitespacesAndNewlines)
                let preview = String(trimmed.prefix(80))
                let metadata = ChunkMetadata(
                    chunkId: chunkId,
                    modelFilename: modelFilename,
                    nTokens: stats.nTokens,
                    prefillTextPreview: preview,
                    createdAt: Date(),
                    captureNCtx: currentNCtx
                )
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                encoder.dateEncodingStrategy = .iso8601
                let metaData = try encoder.encode(metadata)
                try metaData.write(to: chunkStore.metaURL(chunkId: chunkId), options: .atomic)

                let evicted = (try? chunkStore.registerSavedChunk(metadata)) ?? []
                chunkRamCache.removeAll(evicted)

                if PhoneCacheBlendConfig.enableRamHotStitch,
                   let blob = await llamaContext.captureSequenceState(seqId: 0) {
                    chunkRamCache.put(chunkId: chunkId, stateData: blob, nTokens: stats.nTokens)
                }

                results.append(ChunkCacheOpResult(
                    chunkId: chunkId,
                    cacheHit: false,
                    nTokens: stats.nTokens,
                    prefillMs: stats.prefillMs,
                    saveMs: stats.saveMs,
                    evictedChunkIds: evicted
                ))
            } catch {
                messageLog += "Chunk cache save failed: \(chunkId.prefix(8))… \(error)\n"
            }
        }

        await MainActor.run {
            self.chunkCacheSummary = chunkStore.summaryLine()
                + (PhoneCacheBlendConfig.enableRamHotStitch ? " · \(chunkRamCache.summaryLine())" : "")
        }
        return results
    }

    private struct LabelCacheOpResult {
        let listIndex: Int
        let chunkId: String
        let cacheHit: Bool
        let nTokens: Int
    }

    /// RAM-only cache for `[n] ` label KV (reused across queries for the same model).
    private func ensureLabelKvCached(listIndex: Int, llamaContext: LlamaContext) async -> LabelCacheOpResult? {
        guard PhoneCacheBlendConfig.enableLabelKvCache,
              let modelFilename = loadedModelFilename else { return nil }

        let chunkId = ChunkStore.labelChunkId(modelFilename: modelFilename, listIndex: listIndex)
        if let cached = labelRamCache.get(chunkId) {
            return LabelCacheOpResult(
                listIndex: listIndex,
                chunkId: chunkId,
                cacheHit: true,
                nTokens: cached.nTokens
            )
        }

        let labelText = Self.formatChunkLabel(index: listIndex)
        do {
            let snap = try await llamaContext.collectLabelKvSnapshot(labelText: labelText)
            labelRamCache.put(chunkId: chunkId, stateData: snap.data, nTokens: snap.nTokens)
            return LabelCacheOpResult(
                listIndex: listIndex,
                chunkId: chunkId,
                cacheHit: false,
                nTokens: snap.nTokens
            )
        } catch {
            messageLog += "Label cache failed [\(listIndex + 1)]: \(error.localizedDescription)\n"
            return nil
        }
    }

    private func ensureLabelsCached(count: Int, llamaContext: LlamaContext) async -> [LabelCacheOpResult] {
        guard PhoneCacheBlendConfig.enableLabelKvCache else { return [] }
        var results: [LabelCacheOpResult] = []
        for idx in 0..<count {
            if let result = await ensureLabelKvCached(listIndex: idx, llamaContext: llamaContext) {
                results.append(result)
            }
        }
        return results
    }

    private func buildPrefixStitchItem(from result: ChunkCacheOpResult?) -> PrefixStitchItem? {
        guard PhoneCacheBlendConfig.enableSystemPrefixCache,
              let result,
              let chunkStore else { return nil }
        let ram = PhoneCacheBlendConfig.enableRamHotStitch
            ? chunkRamCache.get(result.chunkId)?.stateData
            : nil
        return PrefixStitchItem(
            chunkId: result.chunkId,
            binPath: chunkStore.binURL(chunkId: result.chunkId).path,
            nTokens: result.nTokens,
            ramState: ram
        )
    }

    private func buildStitchItems(from cacheResults: [ChunkCacheOpResult]) -> [ChunkStitchItem] {
        guard let chunkStore, let modelFilename = loadedModelFilename else { return [] }
        return cacheResults.enumerated().map { idx, result in
            let ram = PhoneCacheBlendConfig.enableRamHotStitch
                ? chunkRamCache.get(result.chunkId)?.stateData
                : nil
            let labelEntry = PhoneCacheBlendConfig.enableLabelKvCache
                ? labelRamCache.get(ChunkStore.labelChunkId(modelFilename: modelFilename, listIndex: idx))
                : nil
            return ChunkStitchItem(
                chunkId: result.chunkId,
                labelText: Self.formatChunkLabel(index: idx),
                binPath: chunkStore.binURL(chunkId: result.chunkId).path,
                nTokens: result.nTokens,
                ramState: ram,
                labelRamState: labelEntry?.stateData,
                labelNTokens: labelEntry?.nTokens ?? 0
            )
        }
    }

    private func applyStitchRamSnapshots(_ snapshots: [(chunkId: String, data: Data, nTokens: Int)]) {
        guard PhoneCacheBlendConfig.enableRamHotStitch else { return }
        for snap in snapshots {
            chunkRamCache.put(chunkId: snap.chunkId, stateData: snap.data, nTokens: snap.nTokens)
        }
    }

    @discardableResult
    private func stitchFromCache(
        llamaContext: LlamaContext,
        prefixCacheResult: ChunkCacheOpResult?,
        cacheResults: [ChunkCacheOpResult],
        questionSuffix: String,
        mode: String,
        prefillQuestion: Bool = true
    ) async throws -> ChunkStitchResult {
        let prefixItem = buildPrefixStitchItem(from: prefixCacheResult)
        let items = buildStitchItems(from: cacheResults)
        let result = try await llamaContext.completion_init_with_cached_chunks(
            prefixText: PhoneCacheBlendConfig.systemPrefix,
            prefixItem: prefixItem,
            chunkItems: items,
            questionSuffix: questionSuffix,
            mode: mode,
            prefillQuestion: prefillQuestion
        )
        applyStitchRamSnapshots(result.diskSnapshots)
        return result
    }

    private func logCacheLine(_ label: String, result: ChunkCacheOpResult) {
        let idShort = String(result.chunkId.prefix(8))
        if result.cacheHit {
            messageLog += "\(label) HIT  \(idShort)… (\(result.nTokens) tok)\n"
        } else {
            messageLog += "\(label) SAVE \(idShort)… (\(result.nTokens) tok, prefill \(String(format: "%.0f", result.prefillMs ?? 0)) ms, write \(String(format: "%.0f", result.saveMs ?? 0)) ms)\n"
        }
        if !result.evictedChunkIds.isEmpty {
            let ev = result.evictedChunkIds.map { String($0.prefix(8)) }.joined(separator: ", ")
            messageLog += "    FIFO evicted: \(ev)…\n"
        }
    }

    private func logCacheEnsureResults(
        prefix: ChunkCacheOpResult?,
        passages: [ChunkCacheOpResult],
        labels: [LabelCacheOpResult] = []
    ) {
        guard prefix != nil || !passages.isEmpty else { return }
        messageLog += "\n--- Chunk KV (Phase C reuse) ---\n"
        messageLog += "\(chunkStore?.summaryLine() ?? "")\n"
        if let prefix {
            logCacheLine("[prefix]", result: prefix)
        } else if PhoneCacheBlendConfig.enableSystemPrefixCache {
            messageLog += "[prefix] SKIP (cache unavailable; stitch will GPU-prefill)\n"
        }
        for (index, r) in passages.enumerated() {
            logCacheLine("[\(index + 1)]", result: r)
        }
        if PhoneCacheBlendConfig.enableLabelKvCache, !labels.isEmpty {
            messageLog += "(Label KV: RAM-only; stitch merges cached `[n] ` instead of GPU prefill.)\n"
            for r in labels {
                let idShort = String(r.chunkId.prefix(8))
                if r.cacheHit {
                    messageLog += "  [\(r.listIndex + 1)] label HIT  \(idShort)… (\(r.nTokens) tok)\n"
                } else {
                    messageLog += "  [\(r.listIndex + 1)] label SAVE \(idShort)… (\(r.nTokens) tok)\n"
                }
            }
            messageLog += "(RAM labels: \(labelRamCache.summaryLine()))\n"
        }
        messageLog += "(Reuse path: cached prefix + chunks"
        if PhoneCacheBlendConfig.enableQuestionPrefillAfterFuse
            && PhoneCacheBlendConfig.enableCacheBlendFuse {
            messageLog += "; question GPU prefill after fuse"
        } else {
            messageLog += " + fresh question prefill"
        }
        messageLog += ".)\n"
        if PhoneCacheBlendConfig.enableRamHotStitch {
            messageLog += "(RAM-hot stitch: HIT entries skip disk reload when warm in memory.)\n"
        } else {
            messageLog += "(Note: HIT skips GPU prefill in ensure; stitch reloads from disk each query.)\n"
        }
    }

    private func logChunkCacheResults(_ results: [ChunkCacheOpResult]) {
        logCacheEnsureResults(prefix: nil, passages: results)
    }

    // MARK: - Timing log (ensure → stitch → fuse → first token = E2E)

    private struct PhaseTiming {
        let promptTokens: Int
        let ensureMs: Double?
        let stitchMs: Double?
        let fuseMs: Double?
        let questionPostFuseMs: Double?
        let firstTokenMs: Double
        let e2eTtftMs: Double
        let cacheHits: Int?
        let cacheSaves: Int?
        let stitchDetail: String?
        let recoveryPrefillMs: Double?
        let stitchBreakdown: StitchTimingBreakdown?

        var accountedE2EMs: Double {
            (ensureMs ?? 0) + (stitchMs ?? 0) + (fuseMs ?? 0) + (questionPostFuseMs ?? 0) + firstTokenMs
        }

        var timingResidualMs: Double { e2eTtftMs - accountedE2EMs }

        func formattedLog(path: RagInferencePath, isFallbackRecovery: Bool, decodeSummary: String) -> String {
            var lines: [String] = []
            if path == .phoneCacheBlend && !isFallbackRecovery {
                lines.append("--- Timing (PhoneCacheBlend) ---")
                lines.append(String(format: "Prompt tokens:  %d", promptTokens))
                if let ensureMs, let hits = cacheHits, let saves = cacheSaves {
                    lines.append(String(
                        format: "Ensure:         %.1f ms  (%d HIT, %d SAVE — load cached KV or collect+write)",
                        ensureMs, hits, saves
                    ))
                }
                if let stitchMs {
                    let stitchLabel = PhoneCacheBlendConfig.enableQuestionPrefillAfterFuse
                        && PhoneCacheBlendConfig.enableCacheBlendFuse
                        ? "concat prefix/chunks + labels; question after fuse"
                        : "concat prefix/chunks + label + question GPU"
                    let detail = stitchDetail ?? ""
                    lines.append(String(
                        format: "Stitch:         %.1f ms  (%@)%@",
                        stitchMs, stitchLabel, detail.isEmpty ? "" : "  [\(detail)]"
                    ))
                    if let stitchBreakdown {
                        lines.append(stitchBreakdown.formattedLog(stitchTotalMs: stitchMs))
                    }
                }
                if let fuseMs {
                    let suffixNote = PhoneCacheBlendConfig.enableQuestionPrefillAfterFuse
                        ? ", suffix_len=0"
                        : ""
                    lines.append(String(
                        format: "Fuse:           %.1f ms  (HKVD @ layer 1 + partial recompute, ratio %.2f)",
                        fuseMs,
                        PhoneCacheBlendConfig.hkvdRecompRatio
                    ) + suffixNote)
                }
                if let questionPostFuseMs {
                    lines.append(String(
                        format: "Question GPU:   %.1f ms  (post-fuse prefill)",
                        questionPostFuseMs
                    ))
                }
                lines.append(String(
                    format: "First token:    %.1f ms  (sample 1st answer token after fuse)",
                    firstTokenMs
                ))
                lines.append("─────────────────────────────────")
                lines.append(String(format: "E2E TTFT:       %.1f ms  (= ensure + stitch + fuse + question + first token)", e2eTtftMs))
                lines.append(String(
                    format: "  Check sum:    %.1f ms  (residual %+.1f ms)",
                    accountedE2EMs, timingResidualMs
                ))
                if abs(timingResidualMs) > 2.0 {
                    lines.append("  Note: residual is timer overhead / thread scheduling (< few ms expected).")
                }
            } else if isFallbackRecovery {
                lines.append("--- Timing (PhoneCacheBlend — recovery fallback) ---")
                lines.append(String(format: "Prompt tokens:  %d", promptTokens))
                if let ensureMs {
                    lines.append(String(format: "Ensure:         %.1f ms  (reuse failed; time spent before fallback)", ensureMs))
                }
                if let recoveryPrefillMs {
                    lines.append(String(format: "Recovery prefill: %.1f ms  (full GPU pass after reuse error)", recoveryPrefillMs))
                }
                lines.append(String(format: "First token:    %.1f ms", firstTokenMs))
                lines.append("─────────────────────────────────")
                lines.append(String(format: "E2E TTFT:       %.1f ms", e2eTtftMs))
                let sum = (ensureMs ?? 0) + (recoveryPrefillMs ?? 0) + firstTokenMs
                lines.append(String(format: "  Check sum:    %.1f ms  (residual %+.1f ms)", sum, e2eTtftMs - sum))
            } else {
                lines.append("--- Timing (baseline) ---")
                lines.append(String(format: "Prompt tokens:  %d", promptTokens))
                if let recoveryPrefillMs {
                    lines.append(String(
                        format: "Full prefill:   %.1f ms  (single GPU pass, all layers — no cache/fuse)",
                        recoveryPrefillMs
                    ))
                }
                lines.append(String(
                    format: "First token:    %.1f ms  (sample after prefill)",
                    firstTokenMs
                ))
                lines.append("─────────────────────────────────")
                lines.append(String(format: "E2E TTFT:       %.1f ms  (= full prefill + first token)", e2eTtftMs))
                if let recoveryPrefillMs {
                    let sum = recoveryPrefillMs + firstTokenMs
                    lines.append(String(
                        format: "  Check sum:    %.1f ms  (residual %+.1f ms)",
                        sum, e2eTtftMs - sum
                    ))
                }
            }
            lines.append("")
            lines.append(decodeSummary)
            return lines.joined(separator: "\n")
        }
    }

    private func decodeOnlySummary(from metrics: InferenceMetrics) -> String {
        String(format: """
            --- Decode ---
            Tokens: %d
            Time:   %.1f ms
            Speed:  %.1f tok/s
            """,
            metrics.decodeTokens,
            metrics.decodeMs,
            metrics.decodeTps
        )
    }

    private func parsePassages(_ passagesText: String) -> [String] {
        passagesText
            .components(separatedBy: "\n---\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    @discardableResult
    private func runSingleRagQuery(
        passages: [String],
        question: String,
        label: String,
        path: RagInferencePath = .phoneCacheBlend,
        logToMessage: Bool = true,
        preloadedCache: [ChunkCacheOpResult]? = nil
    ) async -> RagQueryResult? {
        guard let llamaContext else {
            if logToMessage {
                messageLog += "Load a model first (Models → Qwen).\n"
            }
            return nil
        }

        if passages.isEmpty {
            if logToMessage {
                messageLog += "Add at least one passage (separate multiple with --- on its own line).\n"
            }
            return nil
        }
        if question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if logToMessage {
                messageLog += "Enter a question.\n"
            }
            return nil
        }

        if logToMessage {
            messageLog += "\n--- \(label) [\(path.rawValue)] ---\n"
            messageLog += "Passages: \(passages.count)\n"
        }

        let queryStartNs = DispatchTime.now().uptimeNanoseconds
        let trimmedQuestion = question.trimmingCharacters(in: .whitespacesAndNewlines)

        var cacheResults: [ChunkCacheOpResult] = []
        var prefixCacheResult: ChunkCacheOpResult?
        var labelCacheResults: [LabelCacheOpResult] = []
        var cacheEnsureEndNs = queryStartNs

        if path == .phoneCacheBlend {
            if let preloadedCache, preloadedCache.count == passages.count {
                cacheResults = preloadedCache
                if PhoneCacheBlendConfig.enableSystemPrefixCache {
                    prefixCacheResult = await ensureSystemPrefixCached(llamaContext: llamaContext)
                }
            } else {
                prefixCacheResult = await ensureSystemPrefixCached(llamaContext: llamaContext)
                cacheResults = await ensurePassagesCached(passages: passages, llamaContext: llamaContext)
            }
            if PhoneCacheBlendConfig.enableLabelKvCache {
                labelCacheResults = await ensureLabelsCached(count: passages.count, llamaContext: llamaContext)
            }
            cacheEnsureEndNs = DispatchTime.now().uptimeNanoseconds
            if logToMessage {
                logCacheEnsureResults(prefix: prefixCacheResult, passages: cacheResults, labels: labelCacheResults)
            }
        }

        let questionSuffix = PhoneCacheBlendConfig.questionPrefix
            + trimmedQuestion
            + PhoneCacheBlendConfig.answerPrefix

        var inferenceMode = path == .standardLlama ? "baseline_full_prefill" : "phase_c_reuse"
        var stitchSucceeded = false
        var stitchRamHits = 0
        var stitchDiskLoads = 0
        var stitchBreakdown: StitchTimingBreakdown?
        var stitchPhaseEndNs = cacheEnsureEndNs
        var fusePhaseEndNs = cacheEnsureEndNs
        var questionPostFusePhaseEndNs = cacheEnsureEndNs
        var questionPostFuseMs: Double?
        var fuseRan = false
        var recoveryPrefillStartNs = queryStartNs
        var recoveryPrefillEndNs = queryStartNs
        var promptTokenCount = 0
        var fallbackReason: String?

        if path == .standardLlama {
            let prompt = Self.buildRagPrompt(passages: passages, question: trimmedQuestion)
            await llamaContext.clear()
            do {
                _ = try await llamaContext.completion_init(text: prompt, mode: "baseline_full_prefill")
                recoveryPrefillEndNs = DispatchTime.now().uptimeNanoseconds
                promptTokenCount = await llamaContext.promptTokenCount()
            } catch {
                messageLog += "Baseline prefill failed: \(error.localizedDescription)\n"
                await llamaContext.clear()
                return nil
            }
        } else {
            do {
                guard cacheResults.count == passages.count else {
                    throw LlamaError.couldNotLoadCachedChunk
                }

                let deferQuestion = PhoneCacheBlendConfig.enableQuestionPrefillAfterFuse
                    && PhoneCacheBlendConfig.enableCacheBlendFuse

                let fullStitchTokens = await llamaContext.buildStitchTokenList(
                    prefixText: PhoneCacheBlendConfig.systemPrefix,
                    passages: passages,
                    questionSuffix: questionSuffix,
                    includeQuestion: true
                )
                let fuseTokens = deferQuestion
                    ? await llamaContext.buildStitchTokenList(
                        prefixText: PhoneCacheBlendConfig.systemPrefix,
                        passages: passages,
                        questionSuffix: questionSuffix,
                        includeQuestion: false
                    )
                    : fullStitchTokens
                promptTokenCount = fullStitchTokens.count
                let suffixTokenCount = UInt32(
                    await llamaContext.tokenizeChunk(questionSuffix).count
                )
                let checkLayer = await llamaContext.defaultHkvdCheckLayer()

                let stitchResult = try await stitchFromCache(
                    llamaContext: llamaContext,
                    prefixCacheResult: prefixCacheResult,
                    cacheResults: cacheResults,
                    questionSuffix: deferQuestion ? "" : questionSuffix,
                    mode: "phase_c_reuse",
                    prefillQuestion: !deferQuestion
                )
                stitchRamHits = stitchResult.ramHits
                stitchDiskLoads = stitchResult.diskLoads
                stitchBreakdown = stitchResult.timing
                stitchPhaseEndNs = DispatchTime.now().uptimeNanoseconds
                await llamaContext.setStitchTokenList(deferQuestion ? fuseTokens : fullStitchTokens)
                stitchSucceeded = true

                func markStitchComplete() {
                    stitchPhaseEndNs = DispatchTime.now().uptimeNanoseconds
                }

                func prefillQuestionAfterFuseIfNeeded() async throws {
                    guard deferQuestion else {
                        questionPostFusePhaseEndNs = fusePhaseEndNs
                        return
                    }
                    let qMs = try await llamaContext.prefillQuestionSuffix(questionSuffix)
                    questionPostFuseMs = qMs
                    stitchBreakdown?.questionPostFuseMs = qMs
                    stitchBreakdown?.questionTokens = Int(suffixTokenCount)
                    questionPostFusePhaseEndNs = DispatchTime.now().uptimeNanoseconds
                    await llamaContext.setStitchTokenList(fullStitchTokens)
                }

                var precomputedIndices: [UInt32]?

                if PhoneCacheBlendConfig.enableHkvdProbe {
                    do {
                        let probe = try await llamaContext.probeHkvdIndices(
                            checkLayer: checkLayer,
                            prefixText: PhoneCacheBlendConfig.systemPrefix,
                            passages: passages,
                            questionSuffix: questionSuffix,
                            suffixTokenCount: suffixTokenCount,
                            recompRatio: PhoneCacheBlendConfig.hkvdRecompRatio
                        )
                        if logToMessage {
                            messageLog += "\n" + probe.summary
                            messageLog += "(Probe cleared KV; re-stitching…)\n"
                        }
                        precomputedIndices = probe.indices
                        let restitch = try await stitchFromCache(
                            llamaContext: llamaContext,
                            prefixCacheResult: prefixCacheResult,
                            cacheResults: cacheResults,
                            questionSuffix: deferQuestion ? "" : questionSuffix,
                            mode: "phase_c_reuse",
                            prefillQuestion: !deferQuestion
                        )
                        stitchBreakdown = restitch.timing
                        await llamaContext.setStitchTokenList(deferQuestion ? fuseTokens : fullStitchTokens)
                        markStitchComplete()
                    } catch {
                        if logToMessage {
                            messageLog += "\nHKVD probe skipped: \(error.localizedDescription)\n"
                        }
                    }
                }

                if PhoneCacheBlendConfig.enableCacheBlendFuse {
                    let fuseSuffixLen: UInt32 = deferQuestion ? 0 : suffixTokenCount
                    let fuseInputTokens = deferQuestion ? fuseTokens : fullStitchTokens
                    do {
                        let fuse = try await llamaContext.cacheBlendFuse(
                            tokens: fuseInputTokens,
                            suffixLen: fuseSuffixLen,
                            checkLayer: checkLayer,
                            recompRatio: PhoneCacheBlendConfig.hkvdRecompRatio,
                            mode: PhoneCacheBlendConfig.cacheBlendFuseMode,
                            impIndices: precomputedIndices,
                            requireSamplingLogits: !deferQuestion
                        )
                        if logToMessage {
                            messageLog += "\n" + fuse.summary
                        }
                        fuseRan = true
                        fusePhaseEndNs = DispatchTime.now().uptimeNanoseconds
                        try await prefillQuestionAfterFuseIfNeeded()
                        inferenceMode = "phase_e_fuse"
                    } catch {
                        if PhoneCacheBlendConfig.cacheBlendFuseMode == .graph {
                            if logToMessage {
                                messageLog += "\nGRAPH fuse failed: \(error.localizedDescription)\n"
                                messageLog += "Retrying TOKEN_RECOMPUTE fuse…\n"
                            }
                            _ = try await stitchFromCache(
                                llamaContext: llamaContext,
                                prefixCacheResult: prefixCacheResult,
                                cacheResults: cacheResults,
                                questionSuffix: deferQuestion ? "" : questionSuffix,
                                mode: "phase_c_reuse",
                                prefillQuestion: !deferQuestion
                            )
                            await llamaContext.setStitchTokenList(deferQuestion ? fuseTokens : fullStitchTokens)
                            markStitchComplete()
                            let fuse = try await llamaContext.cacheBlendFuse(
                                tokens: fuseInputTokens,
                                suffixLen: fuseSuffixLen,
                                checkLayer: checkLayer,
                                recompRatio: PhoneCacheBlendConfig.hkvdRecompRatio,
                                mode: .tokenRecompute,
                                impIndices: precomputedIndices,
                                requireSamplingLogits: !deferQuestion
                            )
                            if logToMessage {
                                messageLog += "\n" + fuse.summary
                            }
                            fuseRan = true
                            fusePhaseEndNs = DispatchTime.now().uptimeNanoseconds
                            try await prefillQuestionAfterFuseIfNeeded()
                            inferenceMode = "phase_e_fuse"
                        } else {
                            if logToMessage {
                                messageLog += "\nCacheBlend fuse failed: \(error.localizedDescription)\n"
                                messageLog += "Continuing with stitched KV (no fuse)…\n"
                            }
                            _ = try await stitchFromCache(
                                llamaContext: llamaContext,
                                prefixCacheResult: prefixCacheResult,
                                cacheResults: cacheResults,
                                questionSuffix: questionSuffix,
                                mode: "phase_c_reuse",
                                prefillQuestion: true
                            )
                            await llamaContext.setStitchTokenList(fullStitchTokens)
                            markStitchComplete()
                            fusePhaseEndNs = stitchPhaseEndNs
                            questionPostFusePhaseEndNs = stitchPhaseEndNs
                            inferenceMode = "phase_c_reuse"
                        }
                    }
                } else {
                    fusePhaseEndNs = stitchPhaseEndNs
                    questionPostFusePhaseEndNs = stitchPhaseEndNs
                }
            } catch {
                let fallbackPrompt = Self.buildRagPrompt(passages: passages, question: trimmedQuestion)
                fallbackReason = error.localizedDescription
                if logToMessage {
                    if stitchSucceeded {
                        messageLog += "Post-stitch error; fallback to full prefill: \(error.localizedDescription)\n"
                    } else {
                        messageLog += "Reuse prefill failed; fallback to full prefill: \(error.localizedDescription)\n"
                    }
                }
                await llamaContext.clear()
                recoveryPrefillStartNs = DispatchTime.now().uptimeNanoseconds
                do {
                    _ = try await llamaContext.completion_init(text: fallbackPrompt, mode: "fallback_full_prefill")
                    recoveryPrefillEndNs = DispatchTime.now().uptimeNanoseconds
                    fusePhaseEndNs = recoveryPrefillEndNs
                    promptTokenCount = await llamaContext.promptTokenCount()
                } catch {
                    messageLog += "Fallback prefill failed: \(error.localizedDescription)\n"
                    await llamaContext.clear()
                    return nil
                }
                inferenceMode = "fallback_full_prefill"
            }
        }

        let decodeStart = DispatchTime.now().uptimeNanoseconds
        if logToMessage {
            messageLog += "\nAnswer: "
        }
        var e2eTtftMs: Double?
        var firstTokenNs: UInt64?
        var answer = ""
        while await !llamaContext.is_done {
            let piece = await llamaContext.completion_loop()
            if e2eTtftMs == nil {
                let now = DispatchTime.now().uptimeNanoseconds
                firstTokenNs = now
                e2eTtftMs = Double(now - queryStartNs) / 1_000_000.0
            }
            answer += piece
            if logToMessage {
                self.messageLog += piece
            }
        }
        let metrics = await llamaContext.finalize_metrics(
            mode: inferenceMode,
            decode_start_ns: decodeStart
        )
        await llamaContext.clear()

        let finalE2eMs = e2eTtftMs ?? metrics.ttftMs
        let tokenNs = firstTokenNs ?? queryStartNs

        let prefixHit = prefixCacheResult?.cacheHit == true ? 1 : 0
        let prefixSave = prefixCacheResult.map { $0.cacheHit ? 0 : 1 } ?? 0
        let passageHits = cacheResults.filter(\.cacheHit).count
        let passageSaves = cacheResults.count - passageHits
        let labelHits = labelCacheResults.filter(\.cacheHit).count
        let labelSaves = labelCacheResults.count - labelHits
        let cacheHits = passageHits + prefixHit + labelHits
        let cacheSaves = passageSaves + prefixSave + labelSaves

        let ensureMs: Double? = path == .phoneCacheBlend
            ? Double(cacheEnsureEndNs - queryStartNs) / 1_000_000.0
            : nil

        let stitchMs: Double?
        let fuseMs: Double?
        let questionPostFuseMsForLog: Double?
        let firstTokenMs: Double
        let recoveryPrefillMs: Double?
        let isFallbackRecovery = inferenceMode == "fallback_full_prefill"

        if path == .standardLlama {
            recoveryPrefillMs = Double(recoveryPrefillEndNs - queryStartNs) / 1_000_000.0
            stitchMs = nil
            fuseMs = nil
            questionPostFuseMsForLog = nil
            firstTokenMs = Double(tokenNs - recoveryPrefillEndNs) / 1_000_000.0
        } else if isFallbackRecovery {
            recoveryPrefillMs = Double(recoveryPrefillEndNs - recoveryPrefillStartNs) / 1_000_000.0
            stitchMs = nil
            fuseMs = nil
            questionPostFuseMsForLog = nil
            firstTokenMs = Double(tokenNs - recoveryPrefillEndNs) / 1_000_000.0
        } else {
            recoveryPrefillMs = nil
            stitchMs = Double(stitchPhaseEndNs - cacheEnsureEndNs) / 1_000_000.0
            fuseMs = fuseRan
                ? Double(fusePhaseEndNs - stitchPhaseEndNs) / 1_000_000.0
                : 0
            questionPostFuseMsForLog = questionPostFuseMs
            firstTokenMs = Double(tokenNs - questionPostFusePhaseEndNs) / 1_000_000.0
        }

        if promptTokenCount == 0 {
            promptTokenCount = metrics.promptTokens
        }

        let prefixNote = (PhoneCacheBlendConfig.enableSystemPrefixCache && prefixCacheResult != nil)
            ? "prefix + "
            : ""
        let stitchDetail: String?
        if path == .phoneCacheBlend {
            if PhoneCacheBlendConfig.enableRamHotStitch {
                stitchDetail = String(
                    format: "RAM %d / disk %d of %@%d chunks",
                    stitchRamHits, stitchDiskLoads, prefixNote, passages.count
                )
            } else {
                stitchDetail = String(format: "disk reload %d chunks", passages.count)
            }
        } else {
            stitchDetail = nil
        }

        let phaseTiming = PhaseTiming(
            promptTokens: promptTokenCount,
            ensureMs: ensureMs,
            stitchMs: stitchMs,
            fuseMs: fuseMs,
            questionPostFuseMs: questionPostFuseMsForLog,
            firstTokenMs: firstTokenMs,
            e2eTtftMs: finalE2eMs,
            cacheHits: path == .phoneCacheBlend ? cacheHits : nil,
            cacheSaves: path == .phoneCacheBlend ? cacheSaves : nil,
            stitchDetail: stitchDetail,
            recoveryPrefillMs: recoveryPrefillMs,
            stitchBreakdown: stitchBreakdown
        )

        if logToMessage {
            if isFallbackRecovery {
                messageLog += "\n(Note: reuse path failed; recovery full prefill timing below.)\n"
            }
            messageLog += "\n" + phaseTiming.formattedLog(
                path: path,
                isFallbackRecovery: isFallbackRecovery,
                decodeSummary: decodeOnlySummary(from: metrics)
            )
            messageLog += "\n"
        }

        return RagQueryResult(
            answer: answer.trimmingCharacters(in: .whitespacesAndNewlines),
            inferenceMode: inferenceMode,
            e2eTtftMs: finalE2eMs,
            promptTokens: promptTokenCount,
            prefillMs: metrics.prefillMs,
            cacheEnsureMs: ensureMs,
            stitchMs: stitchMs,
            fuseMs: fuseMs,
            firstTokenMs: firstTokenMs,
            cacheHits: path == .phoneCacheBlend ? cacheHits : nil,
            cacheSaves: path == .phoneCacheBlend ? cacheSaves : nil,
            passageCacheHits: path == .phoneCacheBlend ? passageHits : nil,
            passageCacheSaves: path == .phoneCacheBlend ? passageSaves : nil,
            stitchRamHits: path == .phoneCacheBlend ? stitchRamHits : nil,
            stitchDiskLoads: path == .phoneCacheBlend ? stitchDiskLoads : nil,
            stitchBreakdown: stitchBreakdown,
            fallbackReason: fallbackReason
        )
    }

    /// Single prompt mode.
    func completeRag(passagesText: String, question: String) async {
        isInferring = true
        let passages = parsePassages(passagesText)
        _ = await runSingleRagQuery(
            passages: passages,
            question: question,
            label: "RAG query #1",
            path: .phoneCacheBlend
        )
        isInferring = false
    }

    /// Pair mode: run two prompts sequentially to observe chunk reuse.
    func completeRagPair(
        passagesText1: String,
        question1: String,
        passagesText2: String,
        question2: String,
        validationHint: String? = nil
    ) async {
        await completeRagSequence(
            queries: [
                (label: "RAG query #1", passagesText: passagesText1, question: question1),
                (label: "RAG query #2", passagesText: passagesText2, question: question2),
            ],
            validationHint: validationHint
        )
    }

    /// Run an ordered list of RAG prompts (reuse validation / multi-query benchmarks).
    func completeRagSequence(
        queries: [(label: String, passagesText: String, question: String)],
        validationHint: String? = nil
    ) async {
        isInferring = true
        if let validationHint, !validationHint.isEmpty {
            messageLog += "\n" + validationHint + "\n"
        }
        for query in queries {
            let passages = parsePassages(query.passagesText)
            _ = await runSingleRagQuery(
                passages: passages,
                question: query.question,
                label: query.label,
                path: .phoneCacheBlend
            )
        }
        isInferring = false
    }

    /// Run one quality query with the chosen inference path.
    func runQualityQuery(
        _ query: QualityTestQuery,
        suite: QualityTestSuiteKind,
        path: RagInferencePath
    ) async {
        isInferring = true
        guard llamaContext != nil else {
            messageLog += "Load a model first (Models → Qwen).\n"
            isInferring = false
            return
        }

        let requiredCtx = suite == .harder
            ? PhoneCacheBlendConfig.nCtxHarder
            : PhoneCacheBlendConfig.nCtxDefault
        do {
            try await ensureContextCapacity(nCtx: requiredCtx)
        } catch {
            messageLog += "Failed to resize context: \(error.localizedDescription)\n"
            isInferring = false
            return
        }

        guard let llamaContext else {
            isInferring = false
            return
        }

        let genLimit = QualityTestSuite.maxGenTokens(for: query, suite: suite)
        await llamaContext.setMaxGenTokens(genLimit)
        let passages = QualityTestSuite.passageTexts(for: query)
        let label = "Quality [\(suite.rawValue)] Q\(query.id): \(query.title)"

        guard let result = await runSingleRagQuery(
            passages: passages,
            question: query.question,
            label: label,
            path: path,
            logToMessage: true
        ) else {
            isInferring = false
            return
        }

        let scored = QualityScorer.score(answer: result.answer, keyPhrases: query.keyPhrases)
        appendQualityScoreLog(query: query, path: path, scored: scored, ttftMs: result.e2eTtftMs)
        await llamaContext.setMaxGenTokens(128)
        isInferring = false
    }

    /// Run all queries in a quality suite — baseline and PhoneCacheBlend side by side.
    func runQualityMatrix(
        suite: QualityTestSuiteKind = .simple,
        paths: [RagInferencePath] = [.standardLlama, .phoneCacheBlend]
    ) async {
        isInferring = true
        guard self.llamaContext != nil, loadedModelPath != nil else {
            messageLog += "Load a model first (Models → Qwen).\n"
            isInferring = false
            return
        }

        do {
            let requiredCtx = suite == .harder
                ? PhoneCacheBlendConfig.nCtxHarder
                : PhoneCacheBlendConfig.nCtxDefault
            try await ensureContextCapacity(nCtx: requiredCtx)
        } catch {
            messageLog += "Failed to resize context: \(error.localizedDescription)\n"
            isInferring = false
            return
        }

        guard let llamaContext = self.llamaContext else {
            isInferring = false
            return
        }

        await llamaContext.clear()

        let allQueries = QualityTestSuite.queries(for: suite)
        let queryCount = allQueries.count
        let defaultGen = suite.defaultMaxGenTokens
        await llamaContext.setMaxGenTokens(defaultGen)

        messageLog += "\n========================================\n"
        messageLog += "  \(suite.logTitle)\n"
        if suite == .stitchProfile {
            messageLog += "  Timing focus: stitch deserialize / memcpy / GPU (each incl. Metal fence)\n"
        } else if suite == .q2WarmReuse {
            messageLog += "  Timing focus: Q2 cold vs warm PCB (ensure HIT on run 2)\n"
        } else {
            messageLog += "  Pass threshold: \(Int(QualityTestSuite.passThreshold * 100))% key phrases\n"
        }
        if suite == .harder {
            messageLog += "  n_ctx: \(PhoneCacheBlendConfig.nCtxHarder) (Harder_test)\n"
        } else {
            messageLog += "  n_ctx: \(PhoneCacheBlendConfig.nCtxDefault)\n"
        }
        if suite == .reuseMax {
            messageLog += "  Order: Q\(QualityTestSuite.reuseMaxQueryIds.map(String.init).joined(separator: ", Q"))\n"
            messageLog += "  Phase 1: baseline (full prefill each query)\n"
            messageLog += "  Phase 2: Clear KV → PhoneCacheBlend sequential (reuse builds)\n"
        }
        if suite == .stitchProfile {
            messageLog += "  Order: Q\(QualityTestSuite.stitchProfileQueryIds.map(String.init).joined(separator: ", Q"))\n"
            messageLog += "  Phase 1: baseline (prefill + E2E TTFT)\n"
            messageLog += "  Phase 2: Clear KV → PCB (ensure + stitch breakdown + fuse + 1st token)\n"
        }
        if suite == .q2WarmReuse {
            messageLog += "  Order: Q2 run 1, Q2 run 2 (same passages + question)\n"
            messageLog += "  Phase 1: baseline Q2 × 2 (full prefill each; no chunk cache)\n"
            messageLog += "  Phase 2: Clear KV → PCB Q2 × 2 (run 2 should be all HIT / warm)\n"
        }
        messageLog += "========================================\n"

        var allRuns: [QualityQueryRun] = []

        if suite.sequentialPathBenchmark {
            allRuns = await runSequentialPathBenchmark(
                queries: allQueries,
                queryCount: queryCount,
                suite: suite,
                paths: paths,
                llamaContext: llamaContext
            )
        } else {
            allRuns = await runInterleavedQualityBenchmark(
                queries: allQueries,
                queryCount: queryCount,
                suite: suite,
                paths: paths,
                llamaContext: llamaContext
            )
        }

        appendQualitySummary(
            allRuns: allRuns,
            allQueries: allQueries,
            queryCount: queryCount,
            paths: paths,
            suite: suite
        )

        await llamaContext.setMaxGenTokens(128)
        isInferring = false
    }

    /// Per-query baseline then PCB (original matrix order).
    private func runInterleavedQualityBenchmark(
        queries: [QualityTestQuery],
        queryCount: Int,
        suite: QualityTestSuiteKind,
        paths: [RagInferencePath],
        llamaContext: LlamaContext
    ) async -> [QualityQueryRun] {
        var allRuns: [QualityQueryRun] = []

        for query in queries {
            let genLimit = QualityTestSuite.maxGenTokens(for: query, suite: suite)
            await llamaContext.setMaxGenTokens(genLimit)

            messageLog += "\n--- Query \(query.id)/\(queryCount): \(query.title) ---\n"
            messageLog += "Chunks: \(query.passages.count) (\(query.passages.map { "P\($0.rawValue)" }.joined(separator: ", ")))\n"

            for path in paths {
                if let run = await executeQualityQuery(
                    query: query,
                    path: path,
                    suite: suite,
                    logToMessage: false
                ) {
                    allRuns.append(run)
                    appendQualityRunLine(run: run, path: path, query: query, suite: suite)
                } else {
                    messageLog += "[\(path.rawValue)] FAILED (no result)\n"
                }
            }
        }

        return allRuns
    }

    /// All baseline queries, then Clear KV, then all PCB queries (max reuse).
    private func runSequentialPathBenchmark(
        queries: [QualityTestQuery],
        queryCount: Int,
        suite: QualityTestSuiteKind,
        paths: [RagInferencePath],
        llamaContext: LlamaContext
    ) async -> [QualityQueryRun] {
        var allRuns: [QualityQueryRun] = []
        let orderedPaths: [RagInferencePath]
        if paths.contains(.standardLlama) && paths.contains(.phoneCacheBlend) {
            orderedPaths = [.standardLlama, .phoneCacheBlend]
        } else {
            orderedPaths = paths
        }

        for path in orderedPaths {
            if path == .phoneCacheBlend {
                messageLog += "\n>>> Clear KV before PhoneCacheBlend phase <<<\n"
                clearChunkCache()
            }

            let phaseLabel = path == .standardLlama ? "Phase 1 — Baseline" : "Phase 2 — PhoneCacheBlend"
            messageLog += "\n========== \(phaseLabel) ==========\n"

            for (index, query) in queries.enumerated() {
                let genLimit = QualityTestSuite.maxGenTokens(for: query, suite: suite)
                await llamaContext.setMaxGenTokens(genLimit)

                messageLog += "\n--- Q\(query.id) (\(index + 1)/\(queryCount)): \(query.title) ---\n"
                messageLog += "Chunks: \(query.passages.map { "P\($0.rawValue)" }.joined(separator: ", "))\n"

                if let run = await executeQualityQuery(
                    query: query,
                    path: path,
                    suite: suite,
                    logToMessage: true
                ) {
                    allRuns.append(run)
                    appendQualityRunLine(run: run, path: path, query: query, suite: suite)
                } else {
                    messageLog += "[\(path.rawValue)] FAILED (no result)\n"
                }
            }
        }

        return allRuns
    }

    private func executeQualityQuery(
        query: QualityTestQuery,
        path: RagInferencePath,
        suite: QualityTestSuiteKind,
        logToMessage: Bool
    ) async -> QualityQueryRun? {
        let passages = QualityTestSuite.passageTexts(for: query)

        guard let result = await runSingleRagQuery(
            passages: passages,
            question: query.question,
            label: "Q\(query.id) \(path.rawValue)",
            path: path,
            logToMessage: logToMessage
        ) else {
            return nil
        }

        let scored = QualityScorer.score(answer: result.answer, keyPhrases: query.keyPhrases)
        return QualityQueryRun(
            query: query,
            path: path,
            answer: result.answer,
            score: scored,
            e2eTtftMs: result.e2eTtftMs,
            promptTokens: result.promptTokens,
            prefillMs: result.prefillMs,
            cacheEnsureMs: result.cacheEnsureMs,
            stitchMs: result.stitchMs,
            fuseMs: result.fuseMs,
            firstTokenMs: result.firstTokenMs,
            cacheHits: result.cacheHits,
            cacheSaves: result.cacheSaves,
            stitchBreakdown: result.stitchBreakdown
        )
    }

    private func appendQualityRunLine(
        run: QualityQueryRun,
        path: RagInferencePath,
        query: QualityTestQuery,
        suite: QualityTestSuiteKind = .simple
    ) {
        let tag = path == .standardLlama ? "Baseline" : "PCB     "
        if suite.timingOnlyBenchmark {
            if path == .standardLlama {
                let prefill = run.e2eTtftMs - (run.firstTokenMs ?? 0)
                messageLog += String(
                    format: "[Q%d %@] E2E %.0f ms  prefill %.0f + 1st %.0f ms\n",
                    query.id, tag.trimmingCharacters(in: .whitespaces),
                    run.e2eTtftMs, prefill, run.firstTokenMs ?? 0
                )
            } else {
                let qPostFuse = run.stitchBreakdown?.questionPostFuseMs ?? 0
                messageLog += String(
                    format: "[Q%d %@] E2E %.0f ms  ensure %.0f + stitch %.0f + fuse %.0f + q %.0f + 1st %.0f ms\n",
                    query.id, tag.trimmingCharacters(in: .whitespaces),
                    run.e2eTtftMs,
                    run.cacheEnsureMs ?? 0,
                    run.stitchMs ?? 0,
                    run.fuseMs ?? 0,
                    qPostFuse,
                    run.firstTokenMs ?? 0
                )
                if let bd = run.stitchBreakdown, let stitchMs = run.stitchMs {
                    messageLog += String(
                        format: "  stitch: clear %.0f  deser %.0f  memcpy %.0f  label %.0f  q-stitch %.0f  q-fuse %.0f  (core %.0f / %.0f ms)\n",
                        bd.clearMs, bd.deserializeMs, bd.memcpyMs,
                        bd.labelGpuMs + bd.labelCacheMs, bd.questionGpuMs, bd.questionPostFuseMs,
                        bd.coreMs, stitchMs
                    )
                }
            }
            return
        }

        let status = run.score.passed ? "PASS" : "FAIL"
        var line = String(
            format: "[%@] %@  score %.0f%%  E2E %.0f ms",
            tag,
            status,
            run.score.score * 100,
            run.e2eTtftMs
        )
        if path == .phoneCacheBlend {
            line += String(
                format: "  ensure %.0f + stitch %.0f + fuse %.0f + 1st %.0f ms  (%d HIT, %d SAVE)",
                run.cacheEnsureMs ?? 0,
                run.stitchMs ?? 0,
                run.fuseMs ?? 0,
                run.firstTokenMs ?? 0,
                run.cacheHits ?? 0,
                run.cacheSaves ?? 0
            )
        } else {
            line += String(
                format: "  prefill %.0f + 1st %.0f ms",
                (run.e2eTtftMs - (run.firstTokenMs ?? 0)),
                run.firstTokenMs ?? 0
            )
        }
        messageLog += line + "\n"
        if !run.score.missed.isEmpty {
            messageLog += "  missed: \(run.score.missed.joined(separator: ", "))\n"
        }
    }

    private func appendQualitySummary(
        allRuns: [QualityQueryRun],
        allQueries: [QualityTestQuery],
        queryCount: Int,
        paths: [RagInferencePath],
        suite: QualityTestSuiteKind
    ) {
        messageLog += "\n========================================\n"
        messageLog += "  SUMMARY\n"
        messageLog += "========================================\n"

        for path in paths {
            let runs = allRuns.filter { $0.path == path }
            let passed = runs.filter(\.score.passed).count
            let avgScore = runs.isEmpty
                ? 0
                : runs.map(\.score.score).reduce(0, +) / Double(runs.count)
            let avgTtft = runs.isEmpty
                ? 0
                : runs.map(\.e2eTtftMs).reduce(0, +) / Double(runs.count)
            let tag = path == .standardLlama ? "Standard llama" : "PhoneCacheBlend"
            messageLog += String(
                format: "%@: %d/%d pass  avg score %.0f%%  avg E2E TTFT %.0f ms\n",
                tag, passed, runs.count, avgScore * 100, avgTtft
            )
            if path == .phoneCacheBlend, !runs.isEmpty {
                let avgEnsure = runs.compactMap(\.cacheEnsureMs).reduce(0, +) / Double(runs.count)
                let avgStitch = runs.compactMap(\.stitchMs).reduce(0, +) / Double(runs.count)
                let avgFuse = runs.compactMap(\.fuseMs).reduce(0, +) / Double(runs.count)
                let avgFirst = runs.compactMap(\.firstTokenMs).reduce(0, +) / Double(runs.count)
                messageLog += String(
                    format: "  PCB avg  ensure %.0f + stitch %.0f + fuse %.0f + 1st %.0f = %.0f ms\n",
                    avgEnsure, avgStitch, avgFuse, avgFirst,
                    avgEnsure + avgStitch + avgFuse + avgFirst
                )
            }
        }

        if paths.contains(.standardLlama) && paths.contains(.phoneCacheBlend) {
            if suite != .stitchProfile && suite != .q2WarmReuse {
                appendPerQueryPerformanceTable(allRuns: allRuns, allQueries: allQueries)
            }

            let baseline = allRuns.filter { $0.path == .standardLlama }
            let pcb = allRuns.filter { $0.path == .phoneCacheBlend }
            var pcbWins = 0
            var baselineWins = 0
            for query in allQueries {
                guard let b = baseline.first(where: { $0.query.id == query.id }),
                      let p = pcb.first(where: { $0.query.id == query.id }) else {
                    continue
                }
                if p.score.score > b.score.score { pcbWins += 1 }
                else if b.score.score > p.score.score { baselineWins += 1 }
            }
            let ties = queryCount - pcbWins - baselineWins
            if suite != .stitchProfile && suite != .q2WarmReuse {
                messageLog += "Quality head-to-head: PCB better \(pcbWins), baseline better \(baselineWins), ties \(ties)\n"
            }
        }

        if suite == .stitchProfile {
            appendStitchProfileSummary(allRuns: allRuns, allQueries: allQueries)
        }
        if suite == .q2WarmReuse {
            appendQ2WarmReuseSummary(allRuns: allRuns, allQueries: allQueries)
        }

        messageLog += "\nDone.\n"
    }

    private func appendQ2WarmReuseSummary(
        allRuns: [QualityQueryRun],
        allQueries: [QualityTestQuery]
    ) {
        func avg(_ values: [Double]) -> Double {
            values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
        }

        messageLog += "\n--- Q2 E2E by run (ms) ---\n"
        messageLog += "Path       Run 1   Run 2   Run2/Run1\n"
        for path in [RagInferencePath.standardLlama, .phoneCacheBlend] {
            let runs = allQueries.compactMap { q in
                allRuns.first(where: { $0.path == path && $0.query.id == q.id })
            }
            guard runs.count == 2 else { continue }
            let tag = path == .standardLlama ? "Baseline" : "PCB     "
            let ratio = runs[0].e2eTtftMs > 0 ? runs[1].e2eTtftMs / runs[0].e2eTtftMs : 0
            messageLog += String(
                format: "%@ %7.0f %7.0f %7.2fx\n",
                tag, runs[0].e2eTtftMs, runs[1].e2eTtftMs, ratio
            )
        }

        if let b2 = allRuns.first(where: { $0.path == .standardLlama && $0.query.id == 202 }),
           let p2 = allRuns.first(where: { $0.path == .phoneCacheBlend && $0.query.id == 202 }) {
            let speedup = p2.e2eTtftMs > 0 ? b2.e2eTtftMs / p2.e2eTtftMs : 0
            messageLog += String(
                format: "\nPCB run 2 vs baseline run 2: %.0f ms vs %.0f ms  (%.2fx)\n",
                p2.e2eTtftMs, b2.e2eTtftMs, speedup
            )
        }

        let pcbRuns = allRuns.filter { $0.path == .phoneCacheBlend }
        if pcbRuns.count == 2 {
            messageLog += "\n--- PCB phase breakdown (run 1 vs run 2) ---\n"
            messageLog += "              Run 1   Run 2\n"
            let r1 = pcbRuns[0]
            let r2 = pcbRuns[1]
            messageLog += String(
                format: "Ensure      %7.0f %7.0f  (%d HIT → %d HIT)\n",
                r1.cacheEnsureMs ?? 0, r2.cacheEnsureMs ?? 0,
                r1.cacheHits ?? 0, r2.cacheHits ?? 0
            )
            messageLog += String(
                format: "Stitch      %7.0f %7.0f\n",
                r1.stitchMs ?? 0, r2.stitchMs ?? 0
            )
            messageLog += String(
                format: "Fuse        %7.0f %7.0f\n",
                r1.fuseMs ?? 0, r2.fuseMs ?? 0
            )
            if let b1 = r1.stitchBreakdown, let b2 = r2.stitchBreakdown {
                messageLog += String(
                    format: "Q post-fuse %7.0f %7.0f\n",
                    b1.questionPostFuseMs, b2.questionPostFuseMs
                )
            }
            messageLog += String(
                format: "1st token   %7.0f %7.0f\n",
                r1.firstTokenMs ?? 0, r2.firstTokenMs ?? 0
            )
            messageLog += String(
                format: "E2E TTFT    %7.0f %7.0f\n",
                r1.e2eTtftMs, r2.e2eTtftMs
            )
        }

        let breakdowns = pcbRuns.compactMap(\.stitchBreakdown)
        if !breakdowns.isEmpty {
            messageLog += "\n--- PCB stitch averages (both runs) ---\n"
            messageLog += String(
                format: "  clear %.0f  deser %.0f  memcpy %.0f  label cache %.0f  q post-fuse %.0f\n",
                avg(breakdowns.map(\.clearMs)),
                avg(breakdowns.map(\.deserializeMs)),
                avg(breakdowns.map(\.memcpyMs)),
                avg(breakdowns.map(\.labelCacheMs)),
                avg(breakdowns.map(\.questionPostFuseMs))
            )
        }
    }

    private func appendStitchProfileSummary(
        allRuns: [QualityQueryRun],
        allQueries: [QualityTestQuery]
    ) {
        messageLog += "\n--- Stitch component averages (PCB) ---\n"
        let pcbRuns = allRuns.filter { $0.path == .phoneCacheBlend }
        guard !pcbRuns.isEmpty else {
            messageLog += "(no PCB runs)\n"
            return
        }

        func avg(_ values: [Double]) -> Double {
            values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
        }

        let breakdowns = pcbRuns.compactMap(\.stitchBreakdown)
        if !breakdowns.isEmpty {
            messageLog += String(
                format: "  clear %.0f  deserialize %.0f  memcpy %.0f\n",
                avg(breakdowns.map(\.clearMs)),
                avg(breakdowns.map(\.deserializeMs)),
                avg(breakdowns.map(\.memcpyMs))
            )
            messageLog += String(
                format: "  label cache %.0f  label GPU %.0f  question stitch %.0f  question post-fuse %.0f  core sum %.0f\n",
                avg(breakdowns.map(\.labelCacheMs)),
                avg(breakdowns.map(\.labelGpuMs)),
                avg(breakdowns.map(\.questionGpuMs)),
                avg(breakdowns.map(\.questionPostFuseMs)),
                avg(breakdowns.map(\.coreMs))
            )
        }

        messageLog += "\n--- E2E comparison (ms) ---\n"
        messageLog += "Q    Title                  Baseline        PCB   Speedup\n"
        for query in allQueries {
            guard let b = allRuns.first(where: { $0.query.id == query.id && $0.path == .standardLlama }),
                  let p = allRuns.first(where: { $0.query.id == query.id && $0.path == .phoneCacheBlend }) else {
                continue
            }
            let speedup = p.e2eTtftMs > 0 ? b.e2eTtftMs / p.e2eTtftMs : 0
            let title = String(query.title.prefix(22)) as NSString
            messageLog += String(
                format: "Q%-3d %-22@ %10.0f %10.0f %7.2fx\n",
                query.id,
                title,
                b.e2eTtftMs,
                p.e2eTtftMs,
                speedup
            )
        }
    }

    private func appendPerQueryPerformanceTable(
        allRuns: [QualityQueryRun],
        allQueries: [QualityTestQuery]
    ) {
        messageLog += "\n--- Per-query performance (E2E TTFT ms) ---\n"
        messageLog += "Q    Title                  Baseline        PCB   Speedup\n"
        for query in allQueries {
            guard let b = allRuns.first(where: { $0.query.id == query.id && $0.path == .standardLlama }),
                  let p = allRuns.first(where: { $0.query.id == query.id && $0.path == .phoneCacheBlend }) else {
                continue
            }
            let speedup = p.e2eTtftMs > 0 ? b.e2eTtftMs / p.e2eTtftMs : 0
            let title = String(query.title.prefix(20))
            messageLog += String(
                format: "Q%-3d %-20@ %10.0f %10.0f %7.2fx\n",
                query.id, title as NSString, b.e2eTtftMs, p.e2eTtftMs, speedup
            )
        }
        let baselineTotal = allQueries.compactMap { q in
            allRuns.first(where: { $0.query.id == q.id && $0.path == .standardLlama })?.e2eTtftMs
        }.reduce(0, +)
        let pcbTotal = allQueries.compactMap { q in
            allRuns.first(where: { $0.query.id == q.id && $0.path == .phoneCacheBlend })?.e2eTtftMs
        }.reduce(0, +)
        if pcbTotal > 0 {
            messageLog += String(
                format: "Σ    %-20@ %10.0f %10.0f %7.2fx\n",
                "total" as NSString, baselineTotal, pcbTotal, baselineTotal / pcbTotal
            )
        }
    }

    private func appendQualityScoreLog(
        query: QualityTestQuery,
        path: RagInferencePath,
        scored: QualityScorer.Result,
        ttftMs: Double
    ) {
        messageLog += "\n--- Quality score [\(path.rawValue)] ---\n"
        messageLog += String(
            format: "%@  %.0f%% (%d/%d key phrases)  TTFT %.0f ms\n",
            scored.passed ? "PASS" : "FAIL",
            scored.score * 100,
            scored.matched.count,
            query.keyPhrases.count,
            ttftMs
        )
        if !scored.missed.isEmpty {
            messageLog += "Missed phrases: \(scored.missed.joined(separator: ", "))\n"
        }
        messageLog += "\nExpected (reference):\n\(query.expectedAnswer)\n"
    }

    func bench() async {
        guard let llamaContext else {
            return
        }

        messageLog += "\nRunning benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1)
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / 1_000_000_000.0
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)
        messageLog += "\(result)\n"
    }

    func clearChunkCache() {
        do {
            try chunkStore?.clearAll()
            chunkRamCache.removeAll()
            labelRamCache.removeAll()
            chunkCacheSummary = chunkStore?.summaryLine() ?? ""
            if PhoneCacheBlendConfig.enableRamHotStitch {
                chunkCacheSummary += " · \(chunkRamCache.summaryLine())"
            }
            messageLog += "Cleared chunk KV cache (disk FIFO + RAM hot + label RAM).\n"
        } catch {
            messageLog += "Failed to clear chunk cache: \(error)\n"
        }
    }

    private func ramKvBlobStats() -> RamKvBlobStats {
        let chunk = chunkRamCache.stats()
        let labels = labelRamCache.stats()
        return RamKvBlobStats(
            chunkEntries: chunk.entryCount,
            chunkBytes: chunk.totalBytes,
            chunkTokens: chunk.totalTokens,
            labelEntries: labels.entryCount,
            labelBytes: labels.totalBytes,
            labelTokens: labels.totalTokens,
            diskEntries: chunkStore?.cachedCount() ?? 0,
            residentBytes: ProcessMemory.residentBytes(),
            availableBytes: ProcessMemory.availableBytes()
        )
    }

    private func logRamStressLine(
        step: Int,
        targetPassages: Int,
        newSaves: Int,
        stepMs: Double,
        stats: RamKvBlobStats
    ) {
        let resident = stats.residentBytes.map { RamStressFormat.bytes($0) } ?? "n/a"
        let available = stats.availableBytes.map { RamStressFormat.bytes($0) } ?? "n/a"
        messageLog += String(
            format: "step %2d  N=%3d  +%d SAVE  %.0f ms  |  chunk %d × %@ avg  labels %@  |  RAM KV %@  disk %d  |  RSS %@  avail %@\n",
            step,
            targetPassages,
            newSaves,
            stepMs,
            stats.chunkEntries,
            RamStressFormat.bytes(stats.avgChunkBytes),
            RamStressFormat.bytes(stats.labelBytes),
            RamStressFormat.bytes(stats.totalRamBytes),
            stats.diskEntries,
            resident,
            available
        )
    }

    private func estimatedProbeTokens(passageCount: Int, passages: [String]) -> Int {
        guard let chunkStore, let modelFilename = loadedModelFilename else { return passageCount * 600 }
        let prefixId = ChunkStore.prefixChunkId(
            modelFilename: modelFilename,
            prefixText: PhoneCacheBlendConfig.systemPrefix
        )
        let prefixTokens = chunkStore.metadata(for: prefixId)?.nTokens ?? 32
        var passageTokens = 0
        for passage in passages.prefix(passageCount) {
            let id = ChunkStore.chunkId(modelFilename: modelFilename, passageContent: passage)
            passageTokens += chunkStore.metadata(for: id)?.nTokens ?? 560
        }
        return prefixTokens + passageCount * 4 + passageTokens + 12
    }

    private func runRamStressStitchProbe(
        llamaContext: LlamaContext,
        passages: [String],
        probePassageCount: Int,
        nCtx: UInt32,
        label: String
    ) async {
        let probePassages = Array(passages.prefix(probePassageCount))
        guard !probePassages.isEmpty else { return }

        let tokenBudget = Int(nCtx) - 64
        let estimated = estimatedProbeTokens(passageCount: probePassageCount, passages: passages)
        if estimated > tokenBudget {
            messageLog += String(
                format: "  probe %@: SKIP (~%d tok > n_ctx budget %d)\n",
                label, estimated, tokenBudget
            )
            return
        }

        let questionSuffix = PhoneCacheBlendConfig.questionPrefix + "ok" + PhoneCacheBlendConfig.answerPrefix
        let stitchTokens = await llamaContext.buildStitchTokenList(
            prefixText: PhoneCacheBlendConfig.systemPrefix,
            passages: probePassages,
            questionSuffix: questionSuffix,
            includeQuestion: true
        )
        let fuseTokens = PhoneCacheBlendConfig.enableQuestionPrefillAfterFuse
            ? await llamaContext.buildStitchTokenList(
                prefixText: PhoneCacheBlendConfig.systemPrefix,
                passages: probePassages,
                questionSuffix: questionSuffix,
                includeQuestion: false
            )
            : stitchTokens
        messageLog += String(
            format: "  probe %@: ~%d tok est, stitch=%d fuse=%d (n_ctx=%d)\n",
            label, estimated, stitchTokens.count, fuseTokens.count, nCtx
        )

        await llamaContext.clear()
        if let probe = await runSingleRagQuery(
            passages: probePassages,
            question: "ok",
            label: "RAM stress stitch probe \(label)",
            path: .phoneCacheBlend,
            logToMessage: false
        ) {
            let probeStats = ramKvBlobStats()
            if let reason = probe.fallbackReason {
                messageLog += String(
                    format: "  probe %@: FALLBACK E2E %.0f ms  ensure %.0f stitch %.0f fuse %.0f  recovery %.0f  |  %@\n",
                    label,
                    probe.e2eTtftMs,
                    probe.cacheEnsureMs ?? 0,
                    probe.stitchMs ?? 0,
                    probe.fuseMs ?? 0,
                    probe.e2eTtftMs - (probe.firstTokenMs ?? 0) - (probe.cacheEnsureMs ?? 0),
                    reason
                )
            } else {
                let hits = probe.cacheHits ?? 0
                let saves = probe.cacheSaves ?? 0
                messageLog += String(
                    format: "  probe %@: E2E %.0f ms  ensure %.0f (%d HIT, %d SAVE incl. labels) stitch %.0f fuse %.0f  RSS %@  avail %@\n",
                    label,
                    probe.e2eTtftMs,
                    probe.cacheEnsureMs ?? 0,
                    hits,
                    saves,
                    probe.stitchMs ?? 0,
                    probe.fuseMs ?? 0,
                    probeStats.residentBytes.map { RamStressFormat.bytes($0) } ?? "n/a",
                    probeStats.availableBytes.map { RamStressFormat.bytes($0) } ?? "n/a"
                )
            }
        } else {
            messageLog += "  probe \(label): FAILED\n"
        }
    }

    /// After cache ramp, reload @ 8192 if needed and run realistic WikiMQA query probes.
    private func runRamStressWikiStitchProbePhase(
        llamaContext: LlamaContext,
        cumulativePassages: [String],
        cachedPassageCount: Int
    ) async {
        let targetNCtx = PhoneCacheBlendConfig.nCtxWikiMQA
        let alreadyAtTarget = await llamaContext.liveNCtx() == targetNCtx

        if alreadyAtTarget {
            messageLog += "\n--- WikiMQA stitch probe phase (n_ctx=\(targetNCtx), already loaded) ---\n"
        } else {
            messageLog += "\n--- WikiMQA stitch probe phase (n_ctx=\(targetNCtx)) ---\n"
            do {
                try await ensureContextCapacity(nCtx: targetNCtx)
            } catch {
                messageLog += "Failed to resize context for stitch probes: \(error.localizedDescription)\n"
                return
            }
        }

        let nCtx = targetNCtx
        let beforeStats = ramKvBlobStats()
        messageLog += String(
            format: "Context ready: RSS %@  avail %@  (cached %d passages, RAM KV %@)\n",
            beforeStats.residentBytes.map { RamStressFormat.bytes($0) } ?? "n/a",
            beforeStats.availableBytes.map { RamStressFormat.bytes($0) } ?? "n/a",
            cachedPassageCount,
            RamStressFormat.bytes(beforeStats.totalRamBytes)
        )

        messageLog += "\nQuery-size probes:\n"
        for n in PhoneCacheBlendConfig.ramStressWikiStitchProbePassages {
            guard n <= cumulativePassages.count else { continue }
            await runRamStressStitchProbe(
                llamaContext: llamaContext,
                passages: cumulativePassages,
                probePassageCount: n,
                nCtx: nCtx,
                label: "\(n) passages (cache \(cachedPassageCount))"
            )
        }

        if cachedPassageCount >= 10 {
            messageLog += "\nFull WikiMQA query (10 passages) under max cache load:\n"
            await runRamStressStitchProbe(
                llamaContext: llamaContext,
                passages: cumulativePassages,
                probePassageCount: 10,
                nCtx: nCtx,
                label: "10-passage query / cache \(cachedPassageCount)"
            )
        }
    }
    @discardableResult
    private func preloadRamStressPassages(
        scale: RamStressScale,
        targetCount: Int,
        llamaContext: LlamaContext
    ) async -> (passages: [String], cachedCount: Int, aborted: Bool) {
        guard targetCount > 0 else { return ([], 0, false) }

        _ = await ensureSystemPrefixCached(llamaContext: llamaContext)
        var cachedCount = 0

        for idx in 0..<targetCount {
            let passage = RamStressPassages.passageText(index: idx, scale: scale)
            messageLog += String(format: "  · saving passage %d/%d… ", idx + 1, targetCount)
            let oneStart = DispatchTime.now().uptimeNanoseconds
            let batch = await ensurePassagesCached(passages: [passage], llamaContext: llamaContext)
            let oneMs = Double(DispatchTime.now().uptimeNanoseconds - oneStart) / 1_000_000
            if let result = batch.first {
                cachedCount += 1
                let snap = ramKvBlobStats()
                messageLog += String(
                    format: "%.0f ms  (%d tok, RAM KV %@, avail %@)\n",
                    oneMs,
                    result.nTokens,
                    RamStressFormat.bytes(snap.totalRamBytes),
                    snap.availableBytes.map { RamStressFormat.bytes($0) } ?? "n/a"
                )
            } else {
                messageLog += String(
                    format: "FAILED after %.0f ms (OOM likely — last safe N≈%d)\n",
                    oneMs, idx
                )
                break
            }
        }

        let aborted = cachedCount < targetCount
        if PhoneCacheBlendConfig.enableLabelKvCache, cachedCount > 0 {
            for idx in 0..<cachedCount {
                _ = await ensureLabelKvCached(listIndex: idx, llamaContext: llamaContext)
            }
        }

        let passages = (0..<cachedCount).map {
            RamStressPassages.passageText(index: $0, scale: scale)
        }
        return (passages, cachedCount, aborted)
    }

    /// Cache N WikiMQA passages @ n_ctx=8192, then run live PCB stitch probes.
    func runRamStressWikiQueryProbeOnly(
        cachePassageCount: Int = PhoneCacheBlendConfig.ramStressWikiQueryProbeCacheDefault
    ) async {
        isInferring = true
        guard let llamaContext else {
            messageLog += "Load a model first (Models → Qwen).\n"
            isInferring = false
            return
        }

        let priorRamCap = chunkRamCache.stats().maxEntries
        ChunkCacheConfig.setStressDiskMaxEntries(PhoneCacheBlendConfig.wikiChunkDiskMaxEntries)
        chunkRamCache.setMaxEntries(PhoneCacheBlendConfig.wikiChunkRamMaxEntries)

        defer {
            ChunkCacheConfig.setStressDiskMaxEntries(nil)
            chunkRamCache.setMaxEntries(priorRamCap)
            isInferring = false
        }

        do {
            try await ensureContextCapacity(nCtx: PhoneCacheBlendConfig.nCtxWikiMQA)
        } catch {
            messageLog += "Failed to resize context: \(error.localizedDescription)\n"
            return
        }

        clearChunkCache()
        await llamaContext.clear()
        await llamaContext.setMaxGenTokens(1)

        let estBlobRam = cachePassageCount * 17 + 14
        messageLog += "\n========================================\n"
        messageLog += "  WikiMQA Query Probe Only\n"
        messageLog += "  n_ctx=\(PhoneCacheBlendConfig.nCtxWikiMQA) throughout\n"
        messageLog += "  Cache limits: disk \(PhoneCacheBlendConfig.wikiChunkDiskMaxEntries), RAM hot \(PhoneCacheBlendConfig.wikiChunkRamMaxEntries)\n"
        messageLog += "  Cache \(cachePassageCount) passages, then 4/8/10-passage PCB probes\n"
        messageLog += String(
            format: "  Est. phase-1 RAM KV ≈ %d MB (%d × ~17 MB + prefix)\n",
            estBlobRam,
            cachePassageCount
        )
        messageLog += "========================================\n"

        let cacheStart = DispatchTime.now().uptimeNanoseconds
        let preload = await preloadRamStressPassages(
            scale: .wikiScale,
            targetCount: cachePassageCount,
            llamaContext: llamaContext
        )
        let cacheMs = Double(DispatchTime.now().uptimeNanoseconds - cacheStart) / 1_000_000
        let cacheStats = ramKvBlobStats()

        messageLog += String(
            format: "\nCached %d/%d passages in %.0f ms  |  RAM KV %@  RSS %@  avail %@\n",
            preload.cachedCount,
            cachePassageCount,
            cacheMs,
            RamStressFormat.bytes(cacheStats.totalRamBytes),
            cacheStats.residentBytes.map { RamStressFormat.bytes($0) } ?? "n/a",
            cacheStats.availableBytes.map { RamStressFormat.bytes($0) } ?? "n/a"
        )

        if preload.aborted {
            messageLog += "Cache preload aborted — skipping stitch probes.\n"
            await llamaContext.setMaxGenTokens(128)
            return
        }
        if preload.cachedCount < 10 {
            messageLog += "Need at least 10 cached passages for WikiMQA query probe.\n"
            await llamaContext.setMaxGenTokens(128)
            return
        }

        await runRamStressWikiStitchProbePhase(
            llamaContext: llamaContext,
            cumulativePassages: preload.passages,
            cachedPassageCount: preload.cachedCount
        )

        let finalStats = ramKvBlobStats()
        messageLog += "\n--- WikiMQA query probe summary ---\n"
        messageLog += "Background cache: \(preload.cachedCount) passages, \(RamStressFormat.bytes(finalStats.totalRamBytes)) KV blobs\n"
        if let rss = finalStats.residentBytes {
            messageLog += "Final RSS: \(RamStressFormat.bytes(rss))\n"
        }
        if let avail = finalStats.availableBytes {
            messageLog += "Final avail: \(RamStressFormat.bytes(avail))\n"
        }
        messageLog += "\nDone.\n"

        try? await ensureContextCapacity(nCtx: PhoneCacheBlendConfig.nCtxDefault)
        await llamaContext.setMaxGenTokens(128)
        chunkCacheSummary = chunkStore?.summaryLine() ?? ""
        if PhoneCacheBlendConfig.enableRamHotStitch {
            chunkCacheSummary += " · \(chunkRamCache.summaryLine())"
        }
    }

    /// WikiMQA blob ceiling @ n_ctx=8192 (cache-only ramp; use result for benchmark limits).
    func runRamStressWikiBlobCeilingBenchmark() async {
        await runRamStressBenchmark(scale: .wikiScale, stitchProbe: false)
    }

    /// WikiMQA benchmark for one path: baseline full prefill or PhoneCacheBlend (disk 1280, RAM 64).
    func runWikiMQABenchmark(maxQueries: Int, path: RagInferencePath) async {
        isInferring = true
        guard loadedModelPath != nil else {
            messageLog += "Load a model first (Models → Qwen).\n"
            isInferring = false
            return
        }

        let priorRamCap = chunkRamCache.stats().maxEntries
        if path == .phoneCacheBlend {
            ChunkCacheConfig.setStressDiskMaxEntries(PhoneCacheBlendConfig.wikiChunkDiskMaxEntries)
            chunkRamCache.setMaxEntries(PhoneCacheBlendConfig.wikiChunkRamMaxEntries)
        }

        defer {
            if path == .phoneCacheBlend {
                ChunkCacheConfig.setStressDiskMaxEntries(nil)
                chunkRamCache.setMaxEntries(priorRamCap)
            }
            isInferring = false
        }

        let examples: [WikiMQAExample]
        do {
            examples = try WikiMQADataset.load()
        } catch {
            messageLog += "WikiMQA dataset load failed: \(error.localizedDescription)\n"
            return
        }

        let queryCount = min(max(1, maxQueries), examples.count)
        let arm: WikiMQABenchmarkArm = path == .standardLlama ? .baseline : .pcb

        do {
            try await ensureContextCapacity(nCtx: PhoneCacheBlendConfig.nCtxWikiMQA)
        } catch {
            messageLog += "Failed to resize context: \(error.localizedDescription)\n"
            return
        }

        messageLog += "Fresh GPU context for WikiMQA @8192 (Metal stability)…\n"
        do {
            try await reloadLlamaContextAtNCtx(PhoneCacheBlendConfig.nCtxWikiMQA)
        } catch {
            messageLog += "GPU context reload failed: \(error.localizedDescription)\n"
            return
        }

        await llamaContext?.setMaxGenTokens(32)

        let overallStart = DispatchTime.now().uptimeNanoseconds

        messageLog += "\n========================================\n"
        messageLog += "  WikiMQA \(arm.rawValue)\n"
        messageLog += "  n_ctx=\(PhoneCacheBlendConfig.nCtxWikiMQA)\n"
        if path == .phoneCacheBlend {
            messageLog += "  Cache: disk \(PhoneCacheBlendConfig.wikiChunkDiskMaxEntries), RAM hot \(PhoneCacheBlendConfig.wikiChunkRamMaxEntries)\n"
        }
        messageLog += "  Queries: \(queryCount)/\(examples.count)\n"
        messageLog += "  Corpus: 1,055 unique passages, 10 per query\n"
        messageLog += "========================================\n\n"

        let stats = await runWikiMQABenchmarkPhase(
            examples: examples,
            queryCount: queryCount,
            path: path
        )

        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - overallStart) / 1_000_000_000
        messageLog += "\n" + stats.formattedSummary(
            maxQueries: queryCount,
            diskCap: PhoneCacheBlendConfig.wikiChunkDiskMaxEntries,
            ramCap: PhoneCacheBlendConfig.wikiChunkRamMaxEntries,
            nCtx: PhoneCacheBlendConfig.nCtxWikiMQA,
            elapsedSec: elapsed
        ) + "\n"
        if path == .phoneCacheBlend, let diskLine = chunkStore?.summaryLine() {
            messageLog += diskLine
            if PhoneCacheBlendConfig.enableRamHotStitch {
                messageLog += " · \(chunkRamCache.summaryLine())"
            }
            messageLog += "\n"
        }

        messageLog += String(
            format: "\nWikiMQA %@ benchmark complete (%.1f min).\n",
            arm == .baseline ? "baseline" : "PCB",
            elapsed / 60
        )
        if let ctx = llamaContext {
            await ctx.setMaxGenTokens(128)
        }
    }

    private func runWikiMQABenchmarkPhase(
        examples: [WikiMQAExample],
        queryCount: Int,
        path: RagInferencePath
    ) async -> WikiMQABenchmarkStats {
        let arm: WikiMQABenchmarkArm = path == .standardLlama ? .baseline : .pcb
        var stats = WikiMQABenchmarkStats(arm: arm)
        let phaseStart = DispatchTime.now().uptimeNanoseconds
        let nCtx = PhoneCacheBlendConfig.nCtxWikiMQA

        for index in 0..<queryCount {
            // Long contiguous @8192 prefills can leave Metal in a bad state; reload between queries.
            if index > 0 {
                do {
                    try await reloadLlamaContextAtNCtx(nCtx)
                } catch {
                    messageLog += "GPU reload failed before Q\(index + 1): \(error.localizedDescription)\n"
                    stats.recordFailure()
                    continue
                }
            }

            guard llamaContext != nil else {
                stats.recordFailure()
                messageLog += String(format: "Q%03d  %@ FAILED (no context)\n", index + 1, arm == .baseline ? "baseline" : "pcb")
                continue
            }

            let example = examples[index]
            let passages = WikiMQADataset.passages(from: example)
            let question = WikiMQADataset.normalizeQuestion(example.question)
            let gold = WikiMQADataset.goldAnswers(from: example)
            let pathLabel = arm == .baseline ? "WikiMQA Q\(index + 1) baseline" : "WikiMQA Q\(index + 1) PCB"

            var result = await runSingleRagQuery(
                passages: passages,
                question: question,
                label: pathLabel,
                path: path,
                logToMessage: false
            )

            if result == nil {
                messageLog += String(format: "Q%03d  %@ failed — reloading GPU and retrying once…\n", index + 1, arm == .baseline ? "baseline" : "pcb")
                do {
                    try await reloadLlamaContextAtNCtx(nCtx)
                    await llamaContext?.setMaxGenTokens(32)
                    result = await runSingleRagQuery(
                        passages: passages,
                        question: question,
                        label: pathLabel + " (retry)",
                        path: path,
                        logToMessage: false
                    )
                } catch {
                    messageLog += "GPU reload on retry failed: \(error.localizedDescription)\n"
                }
            }

            guard let result else {
                stats.recordFailure()
                messageLog += String(format: "Q%03d  %@ FAILED\n", index + 1, arm == .baseline ? "baseline" : "pcb")
                continue
            }

            let f1 = WikiMQAScorer.bestF1(prediction: result.answer, goldAnswers: gold)
            stats.add(result: result, f1: f1)

            let preview = String(result.answer.prefix(48))
                .replacingOccurrences(of: "\n", with: " ")
            messageLog += stats.formattedQueryLine(
                index: index,
                result: result,
                f1: f1,
                answerPreview: preview
            ) + "\n"

            if (index + 1) % 10 == 0 && index + 1 < queryCount {
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - phaseStart) / 1_000_000_000
                messageLog += String(
                    format: "  … checkpoint %d/%d (%.1f min in this phase)\n",
                    index + 1,
                    queryCount,
                    elapsed / 60
                )
            }
        }

        return stats
    }

    /// Minimal WikiMQA PCB debug: cache only 10 passages in RAM, then 4/8/10-passage probes.
    /// Isolates fuse failures from large background cache (64+ passages).
    func runWikiMQADebugProbe() async {
        isInferring = true
        guard let llamaContext else {
            messageLog += "Load a model first (Models → Qwen).\n"
            isInferring = false
            return
        }

        let priorRamCap = chunkRamCache.stats().maxEntries
        let debugRamCap = 12
        let debugDiskCap = 16
        ChunkCacheConfig.setStressDiskMaxEntries(debugDiskCap)
        chunkRamCache.setMaxEntries(debugRamCap)

        defer {
            ChunkCacheConfig.setStressDiskMaxEntries(nil)
            chunkRamCache.setMaxEntries(priorRamCap)
            isInferring = false
        }

        do {
            try await ensureContextCapacity(nCtx: PhoneCacheBlendConfig.nCtxWikiMQA)
        } catch {
            messageLog += "Failed to resize context: \(error.localizedDescription)\n"
            return
        }

        clearChunkCache()
        await llamaContext.clear()
        await llamaContext.setMaxGenTokens(1)

        let passageCount = 10
        messageLog += "\n========================================\n"
        messageLog += "  WikiMQA Debug Probe (minimal cache)\n"
        messageLog += "  n_ctx=\(PhoneCacheBlendConfig.nCtxWikiMQA)\n"
        messageLog += String(format: "  Cache exactly %d passages (RAM hot %d, disk %d)\n", passageCount, debugRamCap, debugDiskCap)
        messageLog += "  Then PCB probes: 4 / 8 / 10 passages\n"
        messageLog += "========================================\n"

        let preload = await preloadRamStressPassages(
            scale: .wikiScale,
            targetCount: passageCount,
            llamaContext: llamaContext
        )
        let cacheStats = ramKvBlobStats()
        messageLog += String(
            format: "\nCached %d/%d  |  RAM KV %@  RSS %@  avail %@\n",
            preload.cachedCount,
            passageCount,
            RamStressFormat.bytes(cacheStats.totalRamBytes),
            cacheStats.residentBytes.map { RamStressFormat.bytes($0) } ?? "n/a",
            cacheStats.availableBytes.map { RamStressFormat.bytes($0) } ?? "n/a"
        )

        guard preload.cachedCount >= 10 else {
            messageLog += "Need 10 passages cached — aborting probes.\n"
            await llamaContext.setMaxGenTokens(128)
            return
        }

        await runRamStressWikiStitchProbePhase(
            llamaContext: llamaContext,
            cumulativePassages: preload.passages,
            cachedPassageCount: preload.cachedCount
        )

        messageLog += "\n--- WikiMQA debug probe done ---\n"
        try? await ensureContextCapacity(nCtx: PhoneCacheBlendConfig.nCtxDefault)
        await llamaContext.setMaxGenTokens(128)
        chunkCacheSummary = chunkStore?.summaryLine() ?? ""
        if PhoneCacheBlendConfig.enableRamHotStitch {
            chunkCacheSummary += " · \(chunkRamCache.summaryLine())"
        }
    }

    private func appendWikiBenchmarkRecommendations(
        peakPassages: Int,
        stats: RamKvBlobStats,
        aborted: Bool
    ) {
        let ramSlots = PhoneCacheBlendConfig.wikiChunkRamMaxEntries
        messageLog += "\n--- WikiMQA @8192 benchmark recommendations ---\n"
        messageLog += String(
            format: "Measured blob ceiling: %d passages (~%@ KV blobs)\n",
            peakPassages,
            RamStressFormat.bytes(stats.totalRamBytes)
        )
        if stats.chunkEntries > 0 {
            messageLog += String(
                format: "Avg blob: %@ (~%d tok/passage)\n",
                RamStressFormat.bytes(stats.avgChunkBytes),
                stats.chunkTokens / stats.chunkEntries
            )
        }
        messageLog += String(
            format: "Recommend chunkRamMaxEntries: %d (wikiChunkRamMaxEntries)\n",
            ramSlots
        )
        messageLog += "Recommend chunkDiskMaxEntries: \(PhoneCacheBlendConfig.wikiChunkDiskMaxEntries) (1,055 unique + margin)\n"
        messageLog += String(
            format: "Set PhoneCacheBlendConfig.wikiBlobCeilingPassages = %d in LlamaState.swift after a stable run\n",
            peakPassages
        )
        if aborted {
            messageLog += "(Ceiling from last successful passage before abort/OOM.)\n"
        }
        messageLog += "n_ctx for WikiMQA 200: \(PhoneCacheBlendConfig.nCtxWikiMQA)\n"
    }

    /// Ramp cumulative passage KV blobs in RAM; logs blob bytes and process memory each step.
    func runRamStressBenchmark(
        scale: RamStressScale = .simple,
        stitchProbe: Bool = false
    ) async {
        isInferring = true
        guard let llamaContext else {
            messageLog += "Load a model first (Models → Qwen).\n"
            isInferring = false
            return
        }

        let priorRamCap = chunkRamCache.stats().maxEntries
        ChunkCacheConfig.setStressDiskMaxEntries(PhoneCacheBlendConfig.ramStressDiskMaxEntries)
        chunkRamCache.setMaxEntries(PhoneCacheBlendConfig.ramStressRamMaxEntries)

        defer {
            ChunkCacheConfig.setStressDiskMaxEntries(nil)
            chunkRamCache.setMaxEntries(priorRamCap)
            isInferring = false
        }

        let stressNCtx = PhoneCacheBlendConfig.ramStressNCtx(scale: scale, stitchProbe: stitchProbe)
        do {
            try await ensureContextCapacity(nCtx: stressNCtx)
        } catch {
            messageLog += "Failed to resize context: \(error.localizedDescription)\n"
            return
        }

        clearChunkCache()
        await llamaContext.clear()
        await llamaContext.setMaxGenTokens(1)

        let rampSteps = PhoneCacheBlendConfig.ramStressRampSteps(for: scale)

        messageLog += "\n========================================\n"
        messageLog += "  RAM Stress Benchmark\n"
        messageLog += "  Scale: \(scale.rawValue)\n"
        messageLog += "  Ramp: cumulative passage KV blobs → RAM hot cache\n"
        messageLog += "  Caps: disk \(PhoneCacheBlendConfig.ramStressDiskMaxEntries), RAM \(PhoneCacheBlendConfig.ramStressRamMaxEntries)\n"
        messageLog += "  n_ctx: \(stressNCtx)\n"
        if stitchProbe {
            messageLog += "  Stitch probe: ON (1-token PCB when prompt fits n_ctx)\n"
        } else {
            messageLog += "  Stitch probe: OFF (cache preload only)\n"
        }
        if scale == .wikiScale {
            messageLog += "  WikiMQA: production n_ctx; expect lower blob ceiling than 2048 ramp (~182).\n"
        }
        messageLog += "========================================\n"
        messageLog += "step  N   saves  ms     | chunk blobs          | RAM KV   disk | RSS      avail\n"
        messageLog += "----  --- -----  ------ | -------------------- | ------------- | -----------------\n"

        var cumulativePassages: [String] = []
        var priorCount = 0
        var stepIndex = 0
        var aborted = false

        for target in rampSteps {
            stepIndex += 1
            while cumulativePassages.count < target {
                let idx = cumulativePassages.count
                cumulativePassages.append(RamStressPassages.passageText(index: idx, scale: scale))
            }

            let stepStart = DispatchTime.now().uptimeNanoseconds
            if priorCount == 0 {
                _ = await ensureSystemPrefixCached(llamaContext: llamaContext)
            }
            let newPassages = Array(cumulativePassages[priorCount..<target])
            messageLog += String(
                format: ">>> step %d start: target N=%d (passages %d→%d)\n",
                stepIndex, target, priorCount + 1, target
            )
            var cacheResults: [ChunkCacheOpResult] = []
            for (offset, passage) in newPassages.enumerated() {
                let globalIndex = priorCount + offset
                let passageNum = globalIndex + 1
                messageLog += String(format: "  · saving passage %d/%d… ", passageNum, target)
                let oneStart = DispatchTime.now().uptimeNanoseconds
                let batch = await ensurePassagesCached(
                    passages: [passage],
                    llamaContext: llamaContext
                )
                let oneMs = Double(DispatchTime.now().uptimeNanoseconds - oneStart) / 1_000_000
                if let result = batch.first {
                    cacheResults.append(result)
                    let snap = ramKvBlobStats()
                    messageLog += String(
                        format: "%.0f ms  (%d tok, RAM KV %@, avail %@)\n",
                        oneMs,
                        result.nTokens,
                        RamStressFormat.bytes(snap.totalRamBytes),
                        snap.availableBytes.map { RamStressFormat.bytes($0) } ?? "n/a"
                    )
                } else {
                    messageLog += String(
                        format: "FAILED after %.0f ms (OOM likely — last safe N≈%d)\n",
                        oneMs, globalIndex
                    )
                    aborted = true
                    break
                }
            }
            if aborted {
                priorCount += cacheResults.count
                break
            }
            if PhoneCacheBlendConfig.enableLabelKvCache {
                for idx in priorCount..<target {
                    _ = await ensureLabelKvCached(listIndex: idx, llamaContext: llamaContext)
                }
            }
            let stepMs = Double(DispatchTime.now().uptimeNanoseconds - stepStart) / 1_000_000

            let newSaves = cacheResults.filter { !$0.cacheHit }.count
            let stats = ramKvBlobStats()
            logRamStressLine(
                step: stepIndex,
                targetPassages: target,
                newSaves: newSaves,
                stepMs: stepMs,
                stats: stats
            )

            if cacheResults.count < newPassages.count {
                messageLog += "  ABORT: only \(cacheResults.count)/\(newPassages.count) new passages cached (prefill/save failed).\n"
                aborted = true
                break
            }

            if stitchProbe {
                let probeN = scale == .wikiScale
                    ? min(target, PhoneCacheBlendConfig.ramStressWikiStitchProbePassages.max() ?? 10)
                    : target
                if probeN == target || target <= 10 {
                    await runRamStressStitchProbe(
                        llamaContext: llamaContext,
                        passages: cumulativePassages,
                        probePassageCount: probeN,
                        nCtx: stressNCtx,
                        label: scale == .wikiScale ? "N=\(probeN) (cache \(target))" : "N=\(target)"
                    )
                }
            }

            priorCount = target
            await Task.yield()
        }

        if stitchProbe, scale == .wikiScale, !cumulativePassages.isEmpty, priorCount >= 10 {
            messageLog += "\nFull 10-passage WikiMQA query probe:\n"
            await runRamStressStitchProbe(
                llamaContext: llamaContext,
                passages: cumulativePassages,
                probePassageCount: 10,
                nCtx: stressNCtx,
                label: "10-passage query / cache \(priorCount)"
            )
        }

        let finalStats = ramKvBlobStats()
        messageLog += "\n--- RAM stress summary ---\n"
        messageLog += "Scale: \(scale.rawValue)\n"
        messageLog += "Steps completed: \(stepIndex)\n"
        messageLog += "Peak passages cached: \(priorCount)\n"
        messageLog += "Chunk RAM: \(finalStats.chunkEntries) blobs, \(RamStressFormat.bytes(finalStats.chunkBytes))"
        if finalStats.chunkEntries > 0 {
            messageLog += String(
                format: " (avg %@/passage, ~%d tok/passage)",
                RamStressFormat.bytes(finalStats.avgChunkBytes),
                finalStats.chunkTokens / finalStats.chunkEntries
            )
        }
        messageLog += "\n"
        messageLog += "Label RAM: \(RamStressFormat.bytes(finalStats.labelBytes)) (\(finalStats.labelEntries) labels)\n"
        messageLog += "Total KV blob RAM: \(RamStressFormat.bytes(finalStats.totalRamBytes))\n"
        messageLog += "Disk FIFO: \(finalStats.diskEntries) entries\n"
        if let rss = finalStats.residentBytes {
            messageLog += "Final RSS: \(RamStressFormat.bytes(rss))\n"
        }
        if let avail = finalStats.availableBytes {
            messageLog += "Final available (jetsam headroom): \(RamStressFormat.bytes(avail))\n"
        }
        if aborted {
            messageLog += "Stopped early due to cache errors.\n"
        } else {
            messageLog += "Ramp finished without cache errors (jetsam may still occur on next step).\n"
        }
        if scale == .wikiScale {
            appendWikiBenchmarkRecommendations(
                peakPassages: priorCount,
                stats: finalStats,
                aborted: aborted
            )
            try? await ensureContextCapacity(nCtx: PhoneCacheBlendConfig.nCtxDefault)
        }
        messageLog += "\nDone.\n"

        await llamaContext.setMaxGenTokens(128)
        chunkCacheSummary = chunkStore?.summaryLine() ?? ""
        if PhoneCacheBlendConfig.enableRamHotStitch {
            chunkCacheSummary += " · \(chunkRamCache.summaryLine())"
        }
    }

    func clear() async {
        guard let llamaContext else {
            messageLog = ""
            return
        }
        await llamaContext.clear()
        messageLog = ""
    }
}
