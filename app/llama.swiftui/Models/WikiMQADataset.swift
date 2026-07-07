import Foundation

struct WikiMQACtx: Codable, Sendable {
    let title: String
    let text: String
}

struct WikiMQAExample: Codable, Sendable {
    let ctxs: [WikiMQACtx]
    let question: String
    /// Gold answers: one or more acceptable strings per query.
    let answers: [[String]]
}

enum WikiMQADataset {
    static let bundleFilename = "wikimqa_s"

    static func load(from bundle: Bundle = .main) throws -> [WikiMQAExample] {
        guard let url = bundle.url(forResource: bundleFilename, withExtension: "json") else {
            throw WikiMQADatasetError.missingBundleFile
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([WikiMQAExample].self, from: data)
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

    static func passageText(from ctx: WikiMQACtx) -> String {
        let title = ctx.title.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = ctx.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if title.isEmpty {
            return text
        }
        return "\(title)\n\n\(text)"
    }

    static func passages(from example: WikiMQAExample) -> [String] {
        example.ctxs.map { passageText(from: $0) }
    }

    static func goldAnswers(from example: WikiMQAExample) -> [String] {
        example.answers.flatMap { $0 }
    }
}

enum WikiMQADatasetError: LocalizedError {
    case missingBundleFile

    var errorDescription: String? {
        switch self {
        case .missingBundleFile:
            return "wikimqa_s.json not found in app bundle (add Resources/wikimqa_s.json)."
        }
    }
}
