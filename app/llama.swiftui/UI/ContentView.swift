import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var copyStatus: String?
    @State private var sharePayload: SharePayload?

    var body: some View {
        NavigationView {
            VStack(spacing: 8) {
                ScrollView(.vertical, showsIndicators: true) {
                    Text(llamaState.messageLog)
                        .font(.system(size: 12, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                        .onTapGesture {
                            UIApplication.shared.sendAction(
                                #selector(UIResponder.resignFirstResponder),
                                to: nil, from: nil, for: nil
                            )
                        }
                }
                .frame(maxHeight: .infinity)
                .border(Color.gray.opacity(0.3), width: 0.5)

                Text(llamaState.chunkCacheSummary)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)

                if let copyStatus {
                    Text(copyStatus)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                }

                HStack {
                    Button("Clear log") {
                        clear()
                    }
                    .disabled(llamaState.isInferring)

                    Button("Clear KV") {
                        llamaState.clearChunkCache()
                    }
                    .disabled(llamaState.isInferring)

                    Button("Copy log") {
                        copyLogToPasteboard()
                    }

                    Button("Share log") {
                        shareLog()
                    }
                }
                .buttonStyle(.bordered)
                .padding(.horizontal)

                List {
                    NavigationLink(destination: DrawerView(llamaState: llamaState)) {
                        Label("Models", systemImage: "arrow.down.circle")
                    }
                    NavigationLink(destination: IntegrationTestView(llamaState: llamaState)) {
                        Label("Integration tests", systemImage: "checkmark.circle")
                    }
                    .disabled(llamaState.isInferring)
                    NavigationLink(destination: BenchmarksView(llamaState: llamaState)) {
                        Label("Benchmarks", systemImage: "chart.bar")
                    }
                    .disabled(llamaState.isInferring)
                    NavigationLink(destination: FuseResidualProbeView(llamaState: llamaState)) {
                        Label("Fuse residual probe", systemImage: "timer")
                    }
                    .disabled(llamaState.isInferring)
                }
                .listStyle(.insetGrouped)
                .frame(height: 240)
            }
            .navigationBarTitle("PhoneCacheBlend", displayMode: .inline)
            .sheet(item: $sharePayload) { payload in
                ActivityView(activityItems: payload.items)
            }
        }
    }

    func clear() {
        copyStatus = nil
        Task {
            await llamaState.clear()
        }
    }

    private func copyLogToPasteboard() {
        let log = llamaState.messageLog
        guard !log.isEmpty else {
            copyStatus = "Log is empty — nothing to copy."
            return
        }
        // Simple assignment — setItems with mixed String/Data was writing an empty pasteboard.
        // Do not read pasteboard back afterward (iOS privacy often returns nil even on success).
        UIPasteboard.general.string = log
        copyStatus = String(format: "Copied %d chars.", log.count)
    }

    private func shareLog() {
        let log = llamaState.messageLog
        guard !log.isEmpty else {
            copyStatus = "Log is empty — nothing to share."
            return
        }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let name = "PhoneCacheBlend-log-\(formatter.string(from: Date())).txt"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(name)
        do {
            try log.write(to: url, atomically: true, encoding: .utf8)
            sharePayload = SharePayload(items: [url])
            copyStatus = String(format: "Sharing %d chars…", log.count)
        } catch {
            sharePayload = SharePayload(items: [log])
            copyStatus = "Sharing log as text…"
        }
    }

    struct SharePayload: Identifiable {
        let id = UUID()
        let items: [Any]
    }

    struct ActivityView: UIViewControllerRepresentable {
        let activityItems: [Any]

        func makeUIViewController(context: Context) -> UIActivityViewController {
            UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
        }

        func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
    }

    struct DrawerView: View {
        @ObservedObject var llamaState: LlamaState
        @State private var showingHelp = false

        func delete(at offsets: IndexSet) {
            offsets.forEach { offset in
                let model = llamaState.downloadedModels[offset]
                let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
                try? FileManager.default.removeItem(at: fileURL)
            }
            llamaState.downloadedModels.remove(atOffsets: offsets)
        }

        func getDocumentsDirectory() -> URL {
            FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        }

        var body: some View {
            List {
                Section(header: Text("Import custom GGUF")) {
                    InputButton(llamaState: llamaState)
                }
                Section(header: Text("Downloaded")) {
                    ForEach(llamaState.downloadedModels) { model in
                        DownloadButton(
                            llamaState: llamaState,
                            modelName: model.name,
                            modelUrl: model.url,
                            filename: model.filename
                        )
                    }
                    .onDelete(perform: delete)
                }
                Section(header: Text("Recommended")) {
                    ForEach(llamaState.undownloadedModels) { model in
                        DownloadButton(
                            llamaState: llamaState,
                            modelName: model.name,
                            modelUrl: model.url,
                            filename: model.filename
                        )
                    }
                }
            }
            .listStyle(.grouped)
            .navigationBarTitle("Models", displayMode: .inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Help") { showingHelp = true }
                }
            }
            .sheet(isPresented: $showingHelp) {
                NavigationView {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("1. Download Qwen2.5-1.5B-Instruct Q4_K_M (~1 GB).")
                        Text("2. Tap Load to activate the model.")
                        Text("3. Integration tests: Simple_test stitch profile (Q1, Q2) and warm reuse (Q2 × 2).")
                        Text("4. Benchmarks: WikiMQA, WikiMQA clean, MuSiQue, and RAM stress probes.")
                        Spacer()
                    }
                    .padding()
                    .navigationTitle("Help")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button("Done") { showingHelp = false }
                        }
                    }
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
