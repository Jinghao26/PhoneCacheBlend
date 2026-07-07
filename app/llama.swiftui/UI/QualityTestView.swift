import SwiftUI

struct QualityTestView: View {
    @ObservedObject var llamaState: LlamaState
    @State private var wikiQueryProbeCacheCount = PhoneCacheBlendConfig.ramStressWikiQueryProbeCacheDefault

    var body: some View {
        VStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Simple_test benchmarks")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Text("Baseline full prefill vs PhoneCacheBlend (ensure → stitch → fuse). Results append to the main log.")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Button("Stitch profile — Q1, Q2 (baseline → PCB)") {
                    Task {
                        await llamaState.runQualityMatrix(suite: .stitchProfile)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(llamaState.isInferring)

                Button("Warm reuse — Q2 × 2 (baseline ×2 → PCB ×2)") {
                    Task {
                        await llamaState.runQualityMatrix(suite: .q2WarmReuse)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(llamaState.isInferring)
            }
            .padding(.horizontal)

            Divider()
                .padding(.horizontal)

            VStack(alignment: .leading, spacing: 8) {
                Text("WikiMQA @ n_ctx=8192")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Text("Wiki-scale passages (~626 tok each). Query probe uses production cache limits (disk 1280, RAM hot 64).")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Button("WikiMQA RAM stress ramp") {
                    Task {
                        await llamaState.runRamStressBenchmark(scale: .wikiScale, stitchProbe: false)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(llamaState.isInferring)

                Picker("Background cache", selection: $wikiQueryProbeCacheCount) {
                    ForEach(PhoneCacheBlendConfig.ramStressWikiQueryProbeCacheSizes, id: \.self) { n in
                        Text("\(n) passages (~\(n * 17) MB)").tag(n)
                    }
                }
                .pickerStyle(.menu)
                .disabled(llamaState.isInferring)

                Button("WikiMQA query probe") {
                    Task {
                        await llamaState.runRamStressWikiQueryProbeOnly(
                            cachePassageCount: wikiQueryProbeCacheCount
                        )
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.blue)
                .disabled(llamaState.isInferring)

                Button("WikiMQA debug probe (10 passages only)") {
                    Task {
                        await llamaState.runWikiMQADebugProbe()
                    }
                }
                .buttonStyle(.bordered)
                .tint(.orange)
                .disabled(llamaState.isInferring)
                Text("Minimal cache: 10 passages in RAM, then 4/8/10-passage PCB. Use this to isolate fuse failures from large background cache.")
                    .font(.caption2)
                    .foregroundColor(.orange)

                Divider()
                    .padding(.vertical, 4)

                Text("WikiMQA benchmark")
                    .font(.caption)
                    .fontWeight(.semibold)
                Text("Run baseline or PCB separately @ n_ctx=8192. PCB uses disk 1280, RAM hot 64.")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                HStack(spacing: 10) {
                    Button("Baseline × 10") {
                        Task {
                            await llamaState.runWikiMQABenchmark(maxQueries: 10, path: .standardLlama)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(llamaState.isInferring)

                    Button("PCB × 10") {
                        Task {
                            await llamaState.runWikiMQABenchmark(maxQueries: 10, path: .phoneCacheBlend)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .disabled(llamaState.isInferring)
                }

                HStack(spacing: 10) {
                    Button("Baseline × 200") {
                        Task {
                            await llamaState.runWikiMQABenchmark(maxQueries: 200, path: .standardLlama)
                        }
                    }
                    .buttonStyle(.bordered)
                    .disabled(llamaState.isInferring)

                    Button("PCB × 200") {
                        Task {
                            await llamaState.runWikiMQABenchmark(maxQueries: 200, path: .phoneCacheBlend)
                        }
                    }
                    .buttonStyle(.bordered)
                    .tint(.green)
                    .disabled(llamaState.isInferring)
                }

                Text("PCB × 200 ~2–4 h; Baseline × 200 ~3–6 h. Keep app foreground and plugged in.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal)

            Spacer()
        }
        .navigationTitle("Quality tests")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct QualityTestView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            QualityTestView(llamaState: LlamaState())
        }
    }
}
