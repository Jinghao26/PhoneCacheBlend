import CryptoKit
import Foundation

/// On-disk chunk KV index with FIFO eviction.
struct ChunkMetadata: Codable, Equatable {
    let chunkId: String
    let modelFilename: String
    let nTokens: Int
    let prefillTextPreview: String
    let createdAt: Date
    /// 3 = adds captureNCtx (KV blobs are only valid for the n_ctx they were collected at).
    let schemaVersion: Int
    /// llama `n_ctx` when this blob was collected; nil = legacy (assume 2048).
    let captureNCtx: UInt32?

    init(
        chunkId: String,
        modelFilename: String,
        nTokens: Int,
        prefillTextPreview: String,
        createdAt: Date,
        schemaVersion: Int = ChunkCacheConfig.schemaVersion,
        captureNCtx: UInt32? = nil
    ) {
        self.chunkId = chunkId
        self.modelFilename = modelFilename
        self.nTokens = nTokens
        self.prefillTextPreview = prefillTextPreview
        self.createdAt = createdAt
        self.schemaVersion = schemaVersion
        self.captureNCtx = captureNCtx
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        chunkId = try c.decode(String.self, forKey: .chunkId)
        modelFilename = try c.decode(String.self, forKey: .modelFilename)
        nTokens = try c.decode(Int.self, forKey: .nTokens)
        prefillTextPreview = try c.decode(String.self, forKey: .prefillTextPreview)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        schemaVersion = try c.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        captureNCtx = try c.decodeIfPresent(UInt32.self, forKey: .captureNCtx)
    }
}

private struct ChunkManifest: Codable {
    /// Oldest chunk_id at index 0, newest at end.
    var fifo: [String]
    var entries: [String: ChunkMetadata]
}

enum ChunkCacheConfig {
    /// Bump when chunk_id / collect format changes (invalidates old .bin files).
    static let schemaVersion = 3
    /// Max number of chunk KV blobs kept on disk (FIFO eviction).
    static var maxEntries: Int {
        stressDiskMaxEntries ?? PhoneCacheBlendConfig.chunkDiskMaxEntries
    }
    static let cacheFolderName = "chunks"
    static let manifestName = "manifest.json"

    /// Raised during RAM stress benchmark so FIFO does not evict before jetsam.
    private static var stressDiskMaxEntries: Int?

    static func setStressDiskMaxEntries(_ value: Int?) {
        stressDiskMaxEntries = value
    }
}

/// Result of ensuring a passage is present in the chunk cache.
struct ChunkCacheOpResult: Sendable {
    let chunkId: String
    let cacheHit: Bool
    let nTokens: Int
    let prefillMs: Double?
    let saveMs: Double?
    let evictedChunkIds: [String]
}

final class ChunkStore {
    private let baseURL: URL
    private let fileManager = FileManager.default

    init(documentsDirectory: URL) {
        baseURL = documentsDirectory.appendingPathComponent(ChunkCacheConfig.cacheFolderName, isDirectory: true)
        try? fileManager.createDirectory(at: baseURL, withIntermediateDirectories: true)
    }

    // MARK: - chunk_id (content-addressed: raw passage body, no list index)

    /// Stable id from model + passage content. Pass **raw** passage text, not `"[n] …"`.
    static func chunkId(modelFilename: String, passageContent: String) -> String {
        let normalized = passageContent.trimmingCharacters(in: .whitespacesAndNewlines)
        let payload = modelFilename + "\n" + normalized
        let digest = SHA256.hash(data: Data(payload.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Body text prefilled when collecting KV (no BOS, no `[n]` label).
    static func prefillText(for passageContent: String) -> String {
        passageContent.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
    }

    private static let systemPrefixCacheMarker = "__rag_system_prefix__"
    private static let listLabelCacheMarker = "__rag_list_label__"

    /// Stable id for the fixed RAG system prefix (BOS + `PhoneCacheBlendConfig.systemPrefix`).
    static func prefixChunkId(modelFilename: String, prefixText: String) -> String {
        let payload = modelFilename + "\n" + systemPrefixCacheMarker + "\n" + prefixText
        let digest = SHA256.hash(data: Data(payload.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Stable id for a list index label (`"[n] "`), reusable across queries (not passage content).
    static func labelChunkId(modelFilename: String, listIndex: Int) -> String {
        let label = "[\(listIndex + 1)] "
        let payload = modelFilename + "\n" + listLabelCacheMarker + "\n" + label
        let digest = SHA256.hash(data: Data(payload.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - paths

    func binURL(chunkId: String) -> URL {
        baseURL.appendingPathComponent("\(chunkId).bin")
    }

    func metaURL(chunkId: String) -> URL {
        baseURL.appendingPathComponent("\(chunkId).json")
    }

    func metadata(for chunkId: String) -> ChunkMetadata? {
        let url = metaURL(chunkId: chunkId)
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(ChunkMetadata.self, from: data)
    }

    func hasValidCache(chunkId: String, forNCtx nCtx: UInt32) -> Bool {
        guard fileManager.fileExists(atPath: binURL(chunkId: chunkId).path) else { return false }
        guard let meta = metadata(for: chunkId) else { return false }
        guard meta.schemaVersion == ChunkCacheConfig.schemaVersion else { return false }
        let captured = meta.captureNCtx ?? PhoneCacheBlendConfig.nCtxDefault
        return captured == nCtx
    }

    func hasValidCache(chunkId: String) -> Bool {
        guard fileManager.fileExists(atPath: binURL(chunkId: chunkId).path) else { return false }
        guard let meta = metadata(for: chunkId) else { return false }
        return meta.schemaVersion == ChunkCacheConfig.schemaVersion
    }

    // MARK: - FIFO manifest

    func cachedCount() -> Int {
        (try? loadManifest())?.fifo.count ?? 0
    }

    func summaryLine() -> String {
        let manifest = (try? loadManifest()) ?? ChunkManifest(fifo: [], entries: [:])
        return "Chunk cache: \(manifest.fifo.count)/\(ChunkCacheConfig.maxEntries) (FIFO)"
    }

    /// Register metadata after a successful save. Evicts oldest entries when over limit.
    @discardableResult
    func registerSavedChunk(_ metadata: ChunkMetadata) throws -> [String] {
        var manifest = try loadManifest()
        manifest.fifo.removeAll { $0 == metadata.chunkId }
        manifest.entries[metadata.chunkId] = metadata
        manifest.fifo.append(metadata.chunkId)

        var evicted: [String] = []
        while manifest.fifo.count > ChunkCacheConfig.maxEntries {
            let oldest = manifest.fifo.removeFirst()
            manifest.entries.removeValue(forKey: oldest)
            deleteFiles(chunkId: oldest)
            evicted.append(oldest)
        }

        try saveManifest(manifest)
        return evicted
    }

    /// Touch existing entry without changing FIFO order (cache hit).
    func touchExisting(chunkId: String) throws {
        var manifest = try loadManifest()
        guard manifest.entries[chunkId] != nil else { return }
        // FIFO: no reorder on hit
        if !manifest.fifo.contains(chunkId) {
            manifest.fifo.append(chunkId)
        }
        try saveManifest(manifest)
    }

    func clearAll() throws {
        if fileManager.fileExists(atPath: baseURL.path) {
            try fileManager.removeItem(at: baseURL)
        }
        try fileManager.createDirectory(at: baseURL, withIntermediateDirectories: true)
    }

    // MARK: - private

    private func manifestURL() -> URL {
        baseURL.appendingPathComponent(ChunkCacheConfig.manifestName)
    }

    private func loadManifest() throws -> ChunkManifest {
        let url = manifestURL()
        guard fileManager.fileExists(atPath: url.path) else {
            return ChunkManifest(fifo: [], entries: [:])
        }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(ChunkManifest.self, from: data)
    }

    private func saveManifest(_ manifest: ChunkManifest) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(manifest)
        try data.write(to: manifestURL(), options: .atomic)
    }

    private func deleteFiles(chunkId: String) {
        try? fileManager.removeItem(at: binURL(chunkId: chunkId))
        try? fileManager.removeItem(at: metaURL(chunkId: chunkId))
    }
}
