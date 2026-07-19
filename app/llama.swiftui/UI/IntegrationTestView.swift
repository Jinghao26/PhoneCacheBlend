import SwiftUI

struct IntegrationTestView: View {
    @ObservedObject var llamaState: LlamaState

    var body: some View {
        List {
            Section {
                Text("Simple_test timing benchmarks from the home harness. Baseline full prefill vs PhoneCacheBlend (ensure → stitch → fuse). Results append to the main log.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section("Stitch profile") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Q1, Q2 — baseline then PCB")
                        .font(.subheadline)
                    Text("Measures stitch/fuse cost on two distinct Simple_test prompts. Fuse log splits C++ phases vs Post sync (Metal drain after SUBSET).")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Button("Run stitch profile") {
                        Task {
                            await llamaState.runQualityMatrix(suite: .stitchProfile)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(llamaState.isInferring)
                }
                .padding(.vertical, 4)
            }

            Section("Warm reuse") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Q2 × 2 — baseline ×2 then PCB ×2")
                        .font(.subheadline)
                    Text("Second PCB run should be all-cache HIT after warm-up on the same passages.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Button("Run warm reuse") {
                        Task {
                            await llamaState.runQualityMatrix(suite: .q2WarmReuse)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(llamaState.isInferring)
                }
                .padding(.vertical, 4)
            }

            Section("Ensure profile") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Q1, 2, 4, 5, 11, 12 — baseline then PCB")
                        .font(.subheadline)
                    Text("Precomputes system prefix + labels [1]…[10] once after Clear KV (timed separately), then runs PCB so per-query ensure is prefix/label HIT. Breaks ensure into lookup / meta / tokenize / prefill / store / register / RAM warm.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Button("Run ensure profile") {
                        Task {
                            await llamaState.runQualityMatrix(suite: .ensureProfile)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(llamaState.isInferring)
                }
                .padding(.vertical, 4)
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Integration tests")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct IntegrationTestView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            IntegrationTestView(llamaState: LlamaState())
        }
    }
}
