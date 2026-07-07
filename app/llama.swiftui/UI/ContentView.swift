import SwiftUI

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var selectedPreset: RagTestPreset = .simple2
    @State private var passagesText = RagTestPreset.simple2.data.passages1
    @State private var questionText = RagTestPreset.simple2.data.question1
    @State private var runPairMode = RagTestPreset.simple2.data.usePairMode
    @State private var passagesText2 = RagTestPreset.simple2.data.passages2 ?? ""
    @State private var questionText2 = RagTestPreset.simple2.data.question2 ?? ""

    private var editorHeight: CGFloat {
        CGFloat(min(220, 72 + selectedPreset.data.chunkCount * 16))
    }

    private var askButtonLabel: String {
        if let seq = selectedPreset.data.sequentialQueries, !seq.isEmpty {
            return "Ask x\(seq.count) (reuse validate)"
        }
        return runPairMode ? "Ask x2 (reuse demo)" : "Ask (reuse prefill)"
    }

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

                VStack(alignment: .leading, spacing: 6) {
                    Picker("Test case", selection: $selectedPreset) {
                        ForEach(RagTestPreset.allCases) { preset in
                            Text(preset.rawValue).tag(preset)
                        }
                    }
                    .pickerStyle(.menu)
                    .disabled(llamaState.isInferring)
                    .onChange(of: selectedPreset) { preset in
                        loadPreset(preset)
                    }

                    Text(selectedPreset.data.description)
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Button("Load test case") {
                        loadPreset(selectedPreset)
                    }
                    .buttonStyle(.bordered)
                    .disabled(llamaState.isInferring)
                }
                .padding(.horizontal)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Passages (separate with ---)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextEditor(text: $passagesText)
                        .frame(height: editorHeight)
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

                Toggle("Run two prompts sequentially (reuse demo)", isOn: $runPairMode)
                    .padding(.horizontal)
                    .disabled(llamaState.isInferring)

                if runPairMode {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Prompt #2 Passages (separate with ---)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        TextEditor(text: $passagesText2)
                            .frame(height: editorHeight)
                            .border(Color.gray.opacity(0.3), width: 0.5)
                            .disabled(llamaState.isInferring)
                    }
                    .padding(.horizontal)

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Prompt #2 Question")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        TextField("Second question", text: $questionText2)
                            .textFieldStyle(.roundedBorder)
                            .disabled(llamaState.isInferring)
                    }
                    .padding(.horizontal)
                }

                HStack {
                    Button(askButtonLabel) {
                        askRag()
                    }
                    .disabled(llamaState.isInferring)

                    Button("Clear log") {
                        clear()
                    }
                    .disabled(llamaState.isInferring)

                    Button("Clear KV") {
                        llamaState.clearChunkCache()
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
                NavigationLink(destination: QualityTestView(llamaState: llamaState)) {
                    Text("Quality tests (Simple + Harder)")
                }
                .disabled(llamaState.isInferring)
                .padding(.bottom, 8)
            }
            .navigationBarTitle("PhoneCacheBlend", displayMode: .inline)
        }
    }

    func loadPreset(_ preset: RagTestPreset) {
        let data = preset.data
        passagesText = data.passages1
        questionText = data.question1
        runPairMode = data.usePairMode
        if let p2 = data.passages2, let q2 = data.question2 {
            passagesText2 = p2
            questionText2 = q2
        }
    }

    func askRag() {
        Task {
            if let seq = selectedPreset.data.sequentialQueries, !seq.isEmpty {
                await llamaState.completeRagSequence(
                    queries: seq.map { ($0.label, $0.passagesText, $0.question) },
                    validationHint: selectedPreset.data.validationHint
                )
            } else if runPairMode {
                await llamaState.completeRagPair(
                    passagesText1: passagesText,
                    question1: questionText,
                    passagesText2: passagesText2,
                    question2: questionText2,
                    validationHint: selectedPreset.data.validationHint
                )
            } else {
                await llamaState.completeRag(passagesText: passagesText, question: questionText)
            }
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
                        Text("3. Pick a test case (2–8 chunks), then Ask x2 for reuse demo.")
                        Text("3b. Swap validate (AB+CDE): Q1 A,B → Q2 B,A → Q3 C,D,E → Q4 E,C,D (expect 2/2 then 3/3 HIT).")
                        Text("4. Chunks share text across presets — run Scaling then Compact without Clear KV to see cross-case HITs.")
                        Text("5. Within Ask x2, prompt #2 usually swaps one chunk (e.g. 5/6 HIT on Scaling).")
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
