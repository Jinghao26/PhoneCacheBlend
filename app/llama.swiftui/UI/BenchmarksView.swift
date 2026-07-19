import SwiftUI

struct BenchmarksView: View {
    @ObservedObject var llamaState: LlamaState
    @State private var wikiQueryProbeCacheCount = PhoneCacheBlendConfig.ramStressWikiQueryProbeCacheDefault

    var body: some View {
        List {
            wikiMQASection(
                title: "WikiMQA (200)",
                footnote: "Original CacheBlend wikimqa_s.json. PCB arms precompute prefix + labels [1]…[10]. No-fuse = modular reuse only (recomp_ratio=0).",
                dataset: .original,
                sampleCount: 10,
                fullCount: 200
            )

            wikiMQASection(
                title: "WikiMQA clean (102)",
                footnote: "Filtered subset (CacheBlend issue #30). Do not compare F1 to paper 200-query numbers. No-fuse isolates stitch-only accuracy.",
                dataset: .clean,
                sampleCount: 10,
                fullCount: 102
            )

            Section {
                Text("Multi-hop QA (150 queries, musique_s.json). PCB / no-fuse precompute prefix + labels [1]…[10].")
                    .font(.caption)
                    .foregroundColor(.secondary)

                benchmarkTripleSection(
                    label: "10 queries",
                    baseline: {
                        Task { await llamaState.runMusiqueBenchmark(maxQueries: 10, path: .standardLlama) }
                    },
                    pcb: {
                        Task { await llamaState.runMusiqueBenchmark(maxQueries: 10, path: .phoneCacheBlend) }
                    },
                    pcbNoFuse: {
                        Task { await llamaState.runMusiqueBenchmark(maxQueries: 10, path: .phoneCacheBlendNoFuse) }
                    }
                )

                benchmarkTripleSection(
                    label: "150 queries (full set)",
                    baseline: {
                        Task { await llamaState.runMusiqueBenchmark(maxQueries: 150, path: .standardLlama) }
                    },
                    pcb: {
                        Task { await llamaState.runMusiqueBenchmark(maxQueries: 150, path: .phoneCacheBlend) }
                    },
                    pcbNoFuse: {
                        Task { await llamaState.runMusiqueBenchmark(maxQueries: 150, path: .phoneCacheBlendNoFuse) }
                    }
                )
            } header: {
                Text("MuSiQue @ n_ctx=8192")
            }

            Section {
                Text("Wiki-scale passages (~626 tok each). Query probe uses production cache limits (disk 1280, RAM hot 64).")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Button("WikiMQA RAM stress ramp") {
                    Task {
                        await llamaState.runRamStressBenchmark(scale: .wikiScale, stitchProbe: false)
                    }
                }
                .disabled(llamaState.isInferring)

                Picker("Background cache", selection: $wikiQueryProbeCacheCount) {
                    ForEach(PhoneCacheBlendConfig.ramStressWikiQueryProbeCacheSizes, id: \.self) { n in
                        Text("\(n) passages (~\(n * 17) MB)").tag(n)
                    }
                }
                .disabled(llamaState.isInferring)

                Button("WikiMQA query probe") {
                    Task {
                        await llamaState.runRamStressWikiQueryProbeOnly(
                            cachePassageCount: wikiQueryProbeCacheCount
                        )
                    }
                }
                .disabled(llamaState.isInferring)

                Button("WikiMQA debug probe (10 passages)") {
                    Task {
                        await llamaState.runWikiMQADebugProbe()
                    }
                }
                .disabled(llamaState.isInferring)

                Button("WikiMQA baseline diagnostics (clean ×10)") {
                    Task {
                        await llamaState.runWikiMQABaselineDiagnostics(dataset: .clean, queryCount: 10)
                    }
                }
                .disabled(llamaState.isInferring)

                Text("Baseline diagnostics: token budget, prefill failure stage (decode vs logits), reload vs sequential, triple-retry on Q1–Q3. Run when baseline fails with 'logits unavailable'.")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Text("Minimal cache: 10 passages in RAM, then 4/8/10-passage PCB. Isolates fuse failures from large background cache.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            } header: {
                Text("RAM stress & probes")
            } footer: {
                Text("Keep app foreground and plugged in for long runs. Compare PCB vs No-fuse F1 to measure fuse contribution.")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Benchmarks")
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder
    private func wikiMQASection(
        title: String,
        footnote: String,
        dataset: WikiMQADatasetVariant,
        sampleCount: Int,
        fullCount: Int
    ) -> some View {
        Section {
            Text("Run each arm separately @ n_ctx=8192. PCB / no-fuse use disk 1280, RAM hot 64.")
                .font(.caption)
                .foregroundColor(.secondary)
            Text(footnote)
                .font(.caption2)
                .foregroundColor(.secondary)

            benchmarkTripleSection(
                label: "\(sampleCount) queries",
                baseline: {
                    Task {
                        await llamaState.runWikiMQABenchmark(
                            maxQueries: sampleCount,
                            path: .standardLlama,
                            dataset: dataset
                        )
                    }
                },
                pcb: {
                    Task {
                        await llamaState.runWikiMQABenchmark(
                            maxQueries: sampleCount,
                            path: .phoneCacheBlend,
                            dataset: dataset
                        )
                    }
                },
                pcbNoFuse: {
                    Task {
                        await llamaState.runWikiMQABenchmark(
                            maxQueries: sampleCount,
                            path: .phoneCacheBlendNoFuse,
                            dataset: dataset
                        )
                    }
                }
            )

            benchmarkTripleSection(
                label: "\(fullCount) queries",
                baseline: {
                    Task {
                        await llamaState.runWikiMQABenchmark(
                            maxQueries: fullCount,
                            path: .standardLlama,
                            dataset: dataset
                        )
                    }
                },
                pcb: {
                    Task {
                        await llamaState.runWikiMQABenchmark(
                            maxQueries: fullCount,
                            path: .phoneCacheBlend,
                            dataset: dataset
                        )
                    }
                },
                pcbNoFuse: {
                    Task {
                        await llamaState.runWikiMQABenchmark(
                            maxQueries: fullCount,
                            path: .phoneCacheBlendNoFuse,
                            dataset: dataset
                        )
                    }
                }
            )
        } header: {
            Text(title)
        }
    }

    @ViewBuilder
    private func benchmarkTripleSection(
        label: String,
        baseline: @escaping () -> Void,
        pcb: @escaping () -> Void,
        pcbNoFuse: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(.subheadline)
            HStack(spacing: 8) {
                Button("Baseline", action: baseline)
                    .frame(maxWidth: .infinity)
                Button("PCB", action: pcb)
                    .frame(maxWidth: .infinity)
                Button("No-fuse", action: pcbNoFuse)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(llamaState.isInferring)
            Text("No-fuse = ensure + stitch only (no CacheBlend fuse).")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
}

struct BenchmarksView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            BenchmarksView(llamaState: LlamaState())
        }
    }
}
