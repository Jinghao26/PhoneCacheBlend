import Foundation

struct MusiqueCtx: Codable, Sendable {
    let title: String
    let text: String
}

struct MusiqueExample: Codable, Sendable {
    let ctxs: [MusiqueCtx]
    let question: String
    /// Gold answers: flat list of acceptable strings (CacheBlend musique_s.json).
    let answers: [String]
}

enum MusiqueDataset {
    static let bundleFilename = "musique_s"

    /// Match CacheBlend `blend_musique.py` prefix_prompt.
    static let systemPrefix = """
        You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..
        Passages:
        """

    /// Match CacheBlend `query_prompt` (question line; answer suffix added separately).
    static let questionPrefix =
        "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion: "

    static let answerPrefix = "\nAnswer:"

    static let benchmarkPrompts = RagBenchmarkPrompts(
        systemPrefix: systemPrefix,
        questionPrefix: questionPrefix,
        answerPrefix: answerPrefix
    )

    static func load(from bundle: Bundle = .main) throws -> [MusiqueExample] {
        guard let url = bundle.url(forResource: bundleFilename, withExtension: "json") else {
            throw MusiqueDatasetError.missingBundleFile
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([MusiqueExample].self, from: data)
    }

    /// Match CacheBlend `normalize_question`.
    static func normalizeQuestion(_ question: String) -> String {
        var q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        if !q.hasSuffix("?") {
            q += "?"
        }
        guard let first = q.first else { return q }
        return first.lowercased() + q.dropFirst()
    }

    static func passageText(from ctx: MusiqueCtx) -> String {
        let title = ctx.title.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = ctx.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if title.isEmpty {
            return text
        }
        return "\(title)\n\n\(text)"
    }

    static func passages(from example: MusiqueExample) -> [String] {
        example.ctxs.map { passageText(from: $0) }
    }

    static func goldAnswers(from example: MusiqueExample) -> [String] {
        example.answers
    }
}

enum MusiqueDatasetError: LocalizedError {
    case missingBundleFile

    var errorDescription: String? {
        switch self {
        case .missingBundleFile:
            return "musique_s.json not found in app bundle (add Resources/musique_s.json)."
        }
    }
}
