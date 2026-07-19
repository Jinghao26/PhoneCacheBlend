import Foundation

/// Two-case diagnostic that attributes fuse "residual" to Metal sync vs logits prime.
///
/// Not a quality / benchmark suite — it only stitches + fuses once per case (no generation).
enum FuseResidualProbe {
    /// Tiny fixed RAG prompt used for both cases.
    static let passages: [String] = [
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris, France.",
    ]
    static let question = "What is the capital of France?"

    enum CaseID: String, CaseIterable {
        /// Fuse without requiring sampling logits → residual should be post-sync (SUBSET GPU drain).
        case gpuDrainOnly = "A"
        /// Fuse requiring sampling logits → shows whether an extra logits decode runs.
        case withLogitsRequired = "B"

        var title: String {
            switch self {
            case .gpuDrainOnly: return "Case A — no logits required"
            case .withLogitsRequired: return "Case B — logits required"
            }
        }

        var detail: String {
            switch self {
            case .gpuDrainOnly:
                return "requireSamplingLogits=false. Post-fuse time should be mostly Metal sync (leftover SUBSET GPU)."
            case .withLogitsRequired:
                return "requireSamplingLogits=true. If logits are missing after sync, an extra decode appears as logits_prime."
            }
        }

        var requireSamplingLogits: Bool {
            switch self {
            case .gpuDrainOnly: return false
            case .withLogitsRequired: return true
            }
        }
    }

    struct CaseResult: Sendable {
        let caseID: CaseID
        let nTokens: UInt32
        let impCount: Int
        let fuseMs: Double
        let timing: FuseTimingBreakdown

        /// Leftover after phases only (historical “mystery residual”); usually ≈ post sync.
        var phaseResidualMs: Double { fuseMs - timing.phaseMs }
        var unexplainedMs: Double { fuseMs - timing.accountedMs }

        var attribution: String {
            let sync = timing.syncMs
            let prime = timing.logitsPrimeMs
            guard unexplainedMs < 5 else {
                return "Unexplained overhead remains — see Sum accounted."
            }
            if sync >= phaseResidualMs * 0.6 && prime < 5 {
                return "Accounted as post-sync (SUBSET / leftover Metal work)."
            }
            if prime >= 20 {
                return String(
                    format: "Accounted as post-sync %.0f ms + logits prime %.0f ms.",
                    sync, prime
                )
            }
            if sync > 5 && prime > 5 {
                return String(
                    format: "Split: post-sync %.0f ms + logits prime %.0f ms.",
                    sync, prime
                )
            }
            return "Timers line up (phases + post sync + logits)."
        }

        func formattedLog() -> String {
            var lines: [String] = []
            lines.append("--- \(caseID.title) ---")
            lines.append(caseID.detail)
            lines.append(String(
                format: "Tokens: %u  |imp|=%d  fuse=%.1f ms  phase-gap=%.1f ms (≈ post-sync)  unexplained=%.1f ms",
                nTokens, impCount, fuseMs, phaseResidualMs, unexplainedMs
            ))
            lines.append(timing.formattedLog(fuseTotalMs: fuseMs))
            lines.append("Verdict: \(attribution)")
            return lines.joined(separator: "\n")
        }
    }

    struct Report: Sendable {
        let cases: [CaseResult]

        func formattedLog() -> String {
            var lines: [String] = []
            lines.append("========================================")
            lines.append("  Fuse residual probe")
            lines.append("  Goal: name the ~residual ms after fuse")
            lines.append("  Prompt: 2 short passages + 1 question (no decode)")
            lines.append("========================================")
            for c in cases {
                lines.append("")
                lines.append(c.formattedLog())
            }
            lines.append("")
            lines.append(summaryVerdict())
            return lines.joined(separator: "\n")
        }

        private func summaryVerdict() -> String {
            guard
                let a = cases.first(where: { $0.caseID == .gpuDrainOnly }),
                let b = cases.first(where: { $0.caseID == .withLogitsRequired })
            else {
                return "Summary: incomplete (need both Case A and B)."
            }

            var lines: [String] = ["--- Summary ---"]
            lines.append(String(
                format: "Case A post-sync=%.1f ms  logits_prime=%.1f ms  phase-gap=%.1f ms",
                a.timing.syncMs, a.timing.logitsPrimeMs, a.phaseResidualMs
            ))
            lines.append(String(
                format: "Case B post-sync=%.1f ms  logits_prime=%.1f ms%@  phase-gap=%.1f ms",
                b.timing.syncMs,
                b.timing.logitsPrimeMs,
                b.timing.logitsPrimed ? " (ran)" : " (skipped)",
                b.phaseResidualMs
            ))

            if a.timing.syncMs >= a.phaseResidualMs * 0.6 && a.timing.logitsPrimeMs < 5 {
                lines.append(
                    "→ Case A: phase gap is Post sync (GPU drain after C++; SUBSET finished after timer)."
                )
            }
            if b.timing.logitsPrimed && b.timing.logitsPrimeMs >= 20 {
                lines.append(
                    "→ Case B: extra logits decode is real cost on top of sync."
                )
            } else if b.timing.logitsPrimeFailed {
                lines.append(
                    "→ Case B: logits missing after fuse and prime FAILED (see seq_rm / Metal)."
                )
            } else if !b.timing.logitsPrimed {
                lines.append(
                    "→ Case B: logits already ready after sync — no second decode; residual is still sync/SUBSET."
                )
            }
            lines.append(
                "Interpretation: residual is not duplicated bookkeeping; it is wall time that was unlabeled."
            )
            return lines.joined(separator: "\n")
        }
    }
}
