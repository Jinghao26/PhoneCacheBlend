import Foundation

/// Word-level F1 (SQuAD-style normalization). Server WikiMQA uses tokenizer token F1;
/// this is comparable for short answers but not identical across tokenizers.
enum WikiMQAScorer {
    static func parseGeneration(_ s: String) -> String {
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        let firstLine = trimmed.components(separatedBy: "\n").first ?? trimmed
        let words = firstLine.split(separator: " ", omittingEmptySubsequences: true)
        guard let head = words.first else { return firstLine }
        let headLower = head.lowercased()
        if headLower.hasPrefix("yes") { return "Yes" }
        if headLower.hasPrefix("no") { return "No" }
        return firstLine
    }

    static func normalizeAnswer(_ s: String) -> [String] {
        let lowered = s.lowercased()
        let noArticles = lowered.replacingOccurrences(
            of: #"\b(a|an|the)\b"#,
            with: " ",
            options: .regularExpression
        )
        let noPunc = noArticles.unicodeScalars.filter { !CharacterSet.punctuationCharacters.contains($0) }
            .map { Character($0) }
        let collapsed = String(noPunc).split(separator: " ", omittingEmptySubsequences: true)
        return collapsed.map(String.init)
    }

    static func tokenF1(prediction: [String], gold: [String]) -> Double {
        if gold.isEmpty || prediction.isEmpty {
            return gold == prediction ? 1.0 : 0.0
        }
        var goldCounts: [String: Int] = [:]
        for tok in gold { goldCounts[tok, default: 0] += 1 }
        var predCounts: [String: Int] = [:]
        for tok in prediction { predCounts[tok, default: 0] += 1 }
        var overlap = 0
        for (tok, predN) in predCounts {
            if let goldN = goldCounts[tok] {
                overlap += min(predN, goldN)
            }
        }
        if overlap == 0 { return 0 }
        let precision = Double(overlap) / Double(prediction.count)
        let recall = Double(overlap) / Double(gold.count)
        return 2 * precision * recall / (precision + recall)
    }

    static func bestF1(prediction: String, goldAnswers: [String]) -> Double {
        let pred = parseGeneration(prediction)
        let predTokens = normalizeAnswer(pred)
        var best = 0.0
        for gold in goldAnswers {
            let goldTokens = normalizeAnswer(gold)
            best = max(best, tokenF1(prediction: predTokens, gold: goldTokens))
        }
        return best
    }
}

enum WikiMQABenchmarkArm: String {
    case baseline = "Baseline (full prefill)"
    case pcb = "PhoneCacheBlend"
}

struct WikiMQABenchmarkStats {
    let arm: WikiMQABenchmarkArm

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

    /// Ensure phase: prefix + passages + labels (disk cache HIT vs SAVE).
    var ensureHits = 0
    var ensureSaves = 0
    /// Passage bodies only (10 per query).
    var passageEnsureHits = 0
    var passageEnsureSaves = 0
    /// Stitch phase: RAM-hot vs disk reload.
    var stitchRamHits = 0
    var stitchDiskLoads = 0

    init(arm: WikiMQABenchmarkArm) {
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

        if arm == .pcb {
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
        lines.append("--- WikiMQA \(arm.rawValue) summary ---")
        lines.append("Queries: \(queriesRun)/\(maxQueries) ok, \(queriesFailed) failed")
        if arm == .pcb {
            lines.append("PCB fallbacks: \(fallbacks)")
        }
        lines.append(String(format: "Wall time: %.1f min", elapsedSec / 60))
        lines.append("Config: n_ctx=\(nCtx)")
        if arm == .pcb {
            lines.append("Cache: disk FIFO=\(diskCap), RAM hot=\(ramCap)")
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
            }
        }
        lines.append(String(format: "Word F1 (mean):      %.3f  (SQuAD-style; server uses tokenizer F1)", meanF1()))
        if arm == .pcb {
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
        case .pcb:
            let ensureH = result.cacheHits ?? 0
            let ensureS = result.cacheSaves ?? 0
            let passH = result.passageCacheHits ?? 0
            let passS = result.passageCacheSaves ?? 0
            let ramH = result.stitchRamHits ?? 0
            let diskL = result.stitchDiskLoads ?? 0
            return String(
                format: "Q%03d  pcb       E2E %6.0f ms  F1 %.2f  ensure %d/%d  pass %d/%d  stitch RAM %d/disk %d  |  %@",
                index + 1,
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

    static func formattedComparison(
        baseline: WikiMQABenchmarkStats,
        pcb: WikiMQABenchmarkStats,
        queryCount: Int
    ) -> String {
        guard baseline.queriesRun > 0, pcb.queriesRun > 0 else {
            return "--- WikiMQA baseline vs PCB ---\n(insufficient results for comparison)\n"
        }
        let speedup = baseline.meanE2eMs() / pcb.meanE2eMs()
        var lines: [String] = []
        lines.append("--- WikiMQA baseline vs PCB (\(queryCount) queries) ---")
        lines.append(String(
            format: "E2E TTFT mean:  baseline %.0f ms  |  PCB %.0f ms  |  speedup %.2fx",
            baseline.meanE2eMs(),
            pcb.meanE2eMs(),
            speedup
        ))
        lines.append(String(
            format: "Word F1 mean:   baseline %.3f  |  PCB %.3f",
            baseline.meanF1(),
            pcb.meanF1()
        ))
        lines.append("(Speedup = baseline E2E ÷ PCB E2E; both include full prompt + first token.)")
        return lines.joined(separator: "\n")
    }
}
