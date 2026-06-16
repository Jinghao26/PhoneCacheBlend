import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

/// Phase A default model — Qwen2.5-1.5B-Instruct Q4_K_M
enum PhoneCacheBlendConfig {
    static let qwenFilename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    static let qwenDownloadURL =
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

    static let systemPrefix = """
        You are a helpful assistant. Answer the question using only the passages below. Be concise.

        Passages:
        """

    static let questionPrefix = "\n\nQuestion: "
    static let answerPrefix = "\nAnswer:"
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    @Published var isInferring = false

    private var llamaContext: LlamaContext?

    init() {
        loadModelsFromDisk()
        loadDefaultModels()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(
                at: documentsURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            )
            for modelURL in modelURLs where modelURL.pathExtension == "gguf" {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                if !downloadedModels.contains(where: { $0.filename == modelURL.lastPathComponent }) {
                    downloadedModels.append(Model(
                        name: modelName,
                        url: "",
                        filename: modelURL.lastPathComponent,
                        status: "downloaded"
                    ))
                }
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        let qwenURL = getDocumentsDirectory().appendingPathComponent(PhoneCacheBlendConfig.qwenFilename)
        if FileManager.default.fileExists(atPath: qwenURL.path) {
            do {
                try loadModel(modelUrl: qwenURL)
            } catch {
                messageLog += "Failed to load Qwen: \(error)\n"
            }
        } else {
            messageLog += "Download Qwen2.5-1.5B Q4_K_M from Models screen.\n"
        }

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if !FileManager.default.fileExists(atPath: fileURL.path) {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                if !undownloadedModels.contains(where: { $0.filename == model.filename }) {
                    undownloadedModels.append(undownloadedModel)
                }
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private let defaultModels: [Model] = [
        Model(
            name: "Qwen2.5-1.5B-Instruct (Q4_K_M, ~1 GiB) ★",
            url: PhoneCacheBlendConfig.qwenDownloadURL,
            filename: PhoneCacheBlendConfig.qwenFilename,
            status: "download"
        ),
    ]

    func loadModel(modelUrl: URL?) throws {
        guard let modelUrl else {
            messageLog += "No model file selected.\n"
            return
        }
        messageLog += "Loading model...\n"
        llamaContext = try LlamaContext.create_context(path: modelUrl.path())
        messageLog += "Loaded: \(modelUrl.lastPathComponent)\n"
        updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
    }

    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.filename == modelName }
    }

    /// Build full RAG prompt: system + passages + question.
    static func buildRagPrompt(passages: [String], question: String) -> String {
        let trimmedPassages = passages
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let body = trimmedPassages.enumerated().map { index, passage in
            "[\(index + 1)] \(passage)"
        }.joined(separator: "\n\n")

        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        return PhoneCacheBlendConfig.systemPrefix
            + body
            + PhoneCacheBlendConfig.questionPrefix
            + q
            + PhoneCacheBlendConfig.answerPrefix
    }

    /// Phase A: full prefill over entire RAG prompt (no chunk KV reuse).
    func completeRag(passagesText: String, question: String) async {
        guard let llamaContext else {
            messageLog += "Load a model first (Models → Qwen).\n"
            return
        }

        let passages = passagesText
            .components(separatedBy: "\n---\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        if passages.isEmpty {
            messageLog += "Add at least one passage (separate multiple with --- on its own line).\n"
            return
        }
        if question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            messageLog += "Enter a question.\n"
            return
        }

        isInferring = true
        let prompt = Self.buildRagPrompt(passages: passages, question: question)

        messageLog += "\n--- RAG query (full prefill) ---\n"
        messageLog += "Passages: \(passages.count)\n"

        await llamaContext.clear()
        await llamaContext.completion_init(text: prompt, mode: "full_prefill_rag")

        let decodeStart = DispatchTime.now().uptimeNanoseconds
        messageLog += "\nAnswer: "

        Task.detached { [weak self] in
            guard let self else { return }
            var answer = ""
            while await !llamaContext.is_done {
                let piece = await llamaContext.completion_loop()
                answer += piece
                await MainActor.run {
                    self.messageLog += piece
                }
            }

            let metrics = await llamaContext.finalize_metrics(
                mode: "full_prefill_rag",
                decode_start_ns: decodeStart
            )
            await llamaContext.clear()

            await MainActor.run {
                self.messageLog += "\n\n\(metrics.summary)\n"
                self.isInferring = false
            }
        }
    }

    func bench() async {
        guard let llamaContext else {
            return
        }

        messageLog += "\nRunning benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1)
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / 1_000_000_000.0
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)
        messageLog += "\(result)\n"
    }

    func clear() async {
        guard let llamaContext else {
            messageLog = ""
            return
        }
        await llamaContext.clear()
        messageLog = ""
    }
}
