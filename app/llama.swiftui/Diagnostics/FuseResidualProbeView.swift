import SwiftUI

struct FuseResidualProbeView: View {
    @ObservedObject var llamaState: LlamaState

    var body: some View {
        List {
            Section {
                Text(
                    """
                    Isolates the fuse residual into named buckets:
                    • Case A — fuse without requiring logits (Metal sync / SUBSET drain)
                    • Case B — fuse requiring logits (shows if an extra logits decode runs)

                    Uses 2 tiny passages + 1 question. Stitch + fuse only (no generation).
                    Results append to the main log.
                    """
                )
                .font(.caption)
                .foregroundColor(.secondary)
            }

            Section("Probe") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Run Case A then Case B")
                        .font(.subheadline)
                    Text("Same prompt both times; only requireSamplingLogits changes.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Button("Run fuse residual probe") {
                        Task {
                            await llamaState.runFuseResidualProbe()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(llamaState.isInferring)
                }
                .padding(.vertical, 4)
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Fuse residual probe")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct FuseResidualProbeView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            FuseResidualProbeView(llamaState: LlamaState())
        }
    }
}
