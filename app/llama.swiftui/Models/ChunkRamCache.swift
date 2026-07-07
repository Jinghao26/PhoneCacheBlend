import Foundation

/// In-memory chunk KV snapshots for RAM-hot stitch (skip disk reload on HIT).
final class ChunkRamCache {
    struct Entry {
        let stateData: Data
        let nTokens: Int
    }

    struct Stats {
        let entryCount: Int
        let maxEntries: Int
        let totalBytes: Int
        let totalTokens: Int

        var avgBytesPerEntry: Int {
            entryCount > 0 ? totalBytes / entryCount : 0
        }
    }

    private var entries: [String: Entry] = [:]
    /// MRU at end; used for eviction when over capacity.
    private var lru: [String] = []
    private var maxEntries: Int

    init(maxEntries: Int = ChunkRamCacheConfig.maxEntries) {
        self.maxEntries = maxEntries
    }

    func setMaxEntries(_ value: Int) {
        maxEntries = value
        while lru.count > maxEntries {
            let oldest = lru.removeFirst()
            entries.removeValue(forKey: oldest)
        }
    }

    func stats() -> Stats {
        let bytes = entries.values.reduce(0) { $0 + $1.stateData.count }
        let tokens = entries.values.reduce(0) { $0 + $1.nTokens }
        return Stats(
            entryCount: entries.count,
            maxEntries: maxEntries,
            totalBytes: bytes,
            totalTokens: tokens
        )
    }

    func get(_ chunkId: String) -> Entry? {
        guard let entry = entries[chunkId] else { return nil }
        touch(chunkId)
        return entry
    }

    func put(chunkId: String, stateData: Data, nTokens: Int) {
        entries[chunkId] = Entry(stateData: stateData, nTokens: nTokens)
        touch(chunkId)
        while lru.count > maxEntries {
            let oldest = lru.removeFirst()
            entries.removeValue(forKey: oldest)
        }
    }

    func remove(_ chunkId: String) {
        entries.removeValue(forKey: chunkId)
        lru.removeAll { $0 == chunkId }
    }

    func removeAll() {
        entries.removeAll()
        lru.removeAll()
    }

    func removeAll(_ chunkIds: [String]) {
        for id in chunkIds {
            remove(id)
        }
    }

    func count() -> Int { entries.count }

    func summaryLine() -> String {
        "RAM hot: \(entries.count)/\(maxEntries)"
    }

    private func touch(_ chunkId: String) {
        lru.removeAll { $0 == chunkId }
        lru.append(chunkId)
    }
}

enum ChunkRamCacheConfig {
    /// Max passage chunk KV blobs kept in RAM (each ~0.5–2 MB depending on passage length).
    static let maxEntries = PhoneCacheBlendConfig.chunkRamMaxEntries
    /// Max `[n] ` label KV snapshots in RAM (~3–4 tokens each). ≥ max passages per query.
    static let maxLabelEntries = 64
}
