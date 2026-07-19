import Foundation

/// Word-level F1 (same scorer as WikiMQA; server MuSiQue uses tokenizer token F1).
enum MusiqueScorer {
    static func bestF1(prediction: String, goldAnswers: [String]) -> Double {
        WikiMQAScorer.bestF1(prediction: prediction, goldAnswers: goldAnswers)
    }
}

enum MusiqueBenchmarkArm: String {
    case baseline = "Baseline (full prefill)"
    case pcb = "PhoneCacheBlend"
    case pcbNoFuse = "PhoneCacheBlend no-fuse (ratio=0)"

    init(path: RagInferencePath) {
        switch path {
        case .standardLlama: self = .baseline
        case .phoneCacheBlend: self = .pcb
        case .phoneCacheBlendNoFuse: self = .pcbNoFuse
        }
    }

    var usesChunkCache: Bool {
        self == .pcb || self == .pcbNoFuse
    }
}

struct MusiqueBenchmarkStats {
    let arm: MusiqueBenchmarkArm

    var queriesRun = 0
    var queriesFailed = 0
    var totalE2eMs = 0.0
    var totalPrefillMs = 0.0
    var totalEnsureMs = 0.0
    var totalStitchMs = 0.0
    var totalFuseMs = 0.0
    var totalFirstTokenMs = 0.0
    var totalF1 = 0.0
    var fallbacks = 0

    var ensureHits = 0
    var ensureSaves = 0
    var passageEnsureHits = 0
    var passageEnsureSaves = 0
    var stitchRamHits = 0
    var stitchDiskLoads = 0

    init(arm: MusiqueBenchmarkArm) {
        self.arm = arm
    }

    mutating func add(result: RagQueryResult, f1: Double) {
        queriesRun += 1
        totalE2eMs += result.e2eTtftMs
        totalPrefillMs += result.prefillMs
        totalEnsureMs += result.cacheEnsureMs ?? 0
        totalStitchMs += result.stitchMs ?? 0
        totalFuseMs += result.fuseMs ?? 0
        totalFirstTokenMs += result.firstTokenMs ?? 0
        totalF1 += f1
        if result.fallbackReason != nil { fallbacks += 1 }

        if arm.usesChunkCache {
            ensureHits += result.cacheHits ?? 0
            ensureSaves += result.cacheSaves ?? 0
            passageEnsureHits += result.passageCacheHits ?? 0
            passageEnsureSaves += result.passageCacheSaves ?? 0
            stitchRamHits += result.stitchRamHits ?? 0
            stitchDiskLoads += result.stitchDiskLoads ?? 0
        }
    }

    mutating func recordFailure() {
        queriesFailed += 1
    }

    private func rate(hits: Int, saves: Int) -> Double {
        let total = hits + saves
        guard total > 0 else { return 0 }
        return 100.0 * Double(hits) / Double(total)
    }

    func meanE2eMs() -> Double {
        queriesRun > 0 ? totalE2eMs / Double(queriesRun) : 0
    }

    func meanF1() -> Double {
        queriesRun > 0 ? totalF1 / Double(queriesRun) : 0
    }

    func ensureDiskHitRatePercent() -> Double {
        rate(hits: ensureHits, saves: ensureSaves)
    }

    func passageEnsureDiskHitRatePercent() -> Double {
        rate(hits: passageEnsureHits, saves: passageEnsureSaves)
    }

    func stitchRamHitRatePercent() -> Double {
        rate(hits: stitchRamHits, saves: stitchDiskLoads)
    }

    func formattedSummary(
        maxQueries: Int,
        diskCap: Int,
        ramCap: Int,
        nCtx: UInt32,
        elapsedSec: Double
    ) -> String {
        var lines: [String] = []
        lines.append("--- MuSiQue \(arm.rawValue) summary ---")
        lines.append("Queries: \(queriesRun)/\(maxQueries) ok, \(queriesFailed) failed")
        if arm.usesChunkCache {
            lines.append("PCB fallbacks: \(fallbacks)")
        }
        lines.append(String(format: "Wall time: %.1f min", elapsedSec / 60))
        lines.append("Config: n_ctx=\(nCtx)")
        if arm.usesChunkCache {
            lines.append("Cache: disk FIFO=\(diskCap), RAM hot=\(ramCap)")
            if arm == .pcbNoFuse {
                lines.append("Fuse: SKIPPED (modular reuse only, recomp_ratio=0)")
            }
        }
        lines.append("")
        lines.append(String(format: "E2E TTFT (mean):     %.0f ms", meanE2eMs()))
        if queriesRun > 0 {
            switch arm {
            case .baseline:
                lines.append(String(
                    format: "  prefill %.0f  1st tok %.0f ms (means)",
                    totalPrefillMs / Double(queriesRun),
                    totalFirstTokenMs / Double(queriesRun)
                ))
            case .pcb:
                lines.append(String(
                    format: "  ensure %.0f  stitch %.0f  fuse %.0f  1st tok %.0f ms (means)",
                    totalEnsureMs / Double(queriesRun),
                    totalStitchMs / Double(queriesRun),
                    totalFuseMs / Double(queriesRun),
                    totalFirstTokenMs / Double(queriesRun)
                ))
            case .pcbNoFuse:
                lines.append(String(
                    format: "  ensure %.0f  stitch %.0f  fuse 0 (skipped)  1st tok %.0f ms (means)",
                    totalEnsureMs / Double(queriesRun),
                    totalStitchMs / Double(queriesRun),
                    totalFirstTokenMs / Double(queriesRun)
                ))
            }
        }
        lines.append(String(format: "Word F1 (mean):      %.3f  (SQuAD-style; server uses tokenizer F1)", meanF1()))
        if arm.usesChunkCache {
            lines.append("")
            lines.append("Cache HIT rates (cumulative over all ensure/stitch ops):")
            lines.append(String(
                format: "  Disk ensure (all):     %.1f%%  (%d HIT / %d SAVE, %d ops)",
                ensureDiskHitRatePercent(),
                ensureHits,
                ensureSaves,
                ensureHits + ensureSaves
            ))
            lines.append(String(
                format: "  Disk ensure (passages): %.1f%%  (%d HIT / %d SAVE, %d ops)",
                passageEnsureDiskHitRatePercent(),
                passageEnsureHits,
                passageEnsureSaves,
                passageEnsureHits + passageEnsureSaves
            ))
            lines.append(String(
                format: "  RAM stitch (chunks):   %.1f%%  (%d RAM HIT / %d disk load, %d ops)",
                stitchRamHitRatePercent(),
                stitchRamHits,
                stitchDiskLoads,
                stitchRamHits + stitchDiskLoads
            ))
        }
        return lines.joined(separator: "\n")
    }

    func formattedQueryLine(index: Int, result: RagQueryResult, f1: Double, answerPreview: String) -> String {
        switch arm {
        case .baseline:
            return String(
                format: "Q%03d  baseline  E2E %6.0f ms  prefill %6.0f  1st tok %4.0f ms  F1 %.2f  |  %@",
                index + 1,
                result.e2eTtftMs,
                result.prefillMs,
                result.firstTokenMs ?? 0,
                f1,
                answerPreview
            )
        case .pcb, .pcbNoFuse:
            let tag = arm == .pcbNoFuse ? "pcb_nofuse" : "pcb      "
            let ensureH = result.cacheHits ?? 0
            let ensureS = result.cacheSaves ?? 0
            let passH = result.passageCacheHits ?? 0
            let passS = result.passageCacheSaves ?? 0
            let ramH = result.stitchRamHits ?? 0
            let diskL = result.stitchDiskLoads ?? 0
            return String(
                format: "Q%03d  %@ E2E %6.0f ms  F1 %.2f  ensure %d/%d  pass %d/%d  stitch RAM %d/disk %d  |  %@",
                index + 1,
                tag,
                result.e2eTtftMs,
                f1,
                ensureH,
                ensureH + ensureS,
                passH,
                passH + passS,
                ramH,
                diskL,
                answerPreview
            )
        }
    }
}
