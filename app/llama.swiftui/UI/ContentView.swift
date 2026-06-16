import SwiftUI

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var passagesText = """
        Hannibal and Scipio is a Caroline era stage play written by Thomas Nabbes. It was first performed in 1635. He was educated at Exeter College, Oxford in 1621.

        ---

        The Punica is a Latin epic poem by Silius Italicus about the Second Punic War between Hannibal and Scipio Africanus.
        """
    @State private var questionText = "Where was the author of Hannibal and Scipio educated?"

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

                VStack(alignment: .leading, spacing: 4) {
                    Text("Passages (separate with ---)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextEditor(text: $passagesText)
                        .frame(height: 100)
                        .border(Color.gray.opacity(0.3), width: 0.5)
                        .disabled(llamaState.isInferring)
                }
                .padding(.horizontal)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Question")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextField("Your question", text: $questionText)
                        .textFieldStyle(.roundedBorder)
                        .disabled(llamaState.isInferring)
                }
                .padding(.horizontal)

                HStack {
                    Button("Ask (full prefill)") {
                        askRag()
                    }
                    .disabled(llamaState.isInferring)

                    Button("Clear") {
                        clear()
                    }
                    .disabled(llamaState.isInferring)

                    Button("Copy log") {
                        UIPasteboard.general.string = llamaState.messageLog
                    }
                }
                .buttonStyle(.bordered)
                .padding(.horizontal)

                NavigationLink(destination: DrawerView(llamaState: llamaState)) {
                    Text("Models (download Qwen)")
                }
                .padding(.bottom, 8)
            }
            .navigationBarTitle("PhoneCacheBlend", displayMode: .inline)
        }
    }

    func askRag() {
        Task {
            await llamaState.completeRag(passagesText: passagesText, question: questionText)
        }
    }

    func clear() {
        Task {
            await llamaState.clear()
        }
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
                        Text("3. Return to main screen and tap Ask (full prefill).")
                        Text("4. Metrics (prefill ms, TTFT) appear in the log.")
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
