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

enum WikiMQADatasetVariant: String, CaseIterable, Identifiable, Sendable {
    /// Original CacheBlend `wikimqa_s.json` (200 queries).
    case original = "WikiMQA (200)"
    /// Filtered subset from CacheBlend issue #30 (102 queries, gold fixes).
    case clean = "WikiMQA clean (102)"

    var id: String { rawValue }

    var bundleFilename: String {
        switch self {
        case .original: return "wikimqa_s"
        case .clean: return "wikimqa_s_clean"
        }
    }

    var logLabel: String {
        switch self {
        case .original: return "WikiMQA"
        case .clean: return "WikiMQA clean"
        }
    }

    var corpusNote: String {
        switch self {
        case .original:
            return "Corpus: 1,055 unique passages, 10 per query"
        case .clean:
            return "Filtered 102-query subset (CacheBlend #30); same ctxs for kept rows"
        }
    }
}

enum WikiMQADataset {
    static func load(
        variant: WikiMQADatasetVariant = .original,
        from bundle: Bundle = .main
    ) throws -> [WikiMQAExample] {
        guard let url = bundle.url(forResource: variant.bundleFilename, withExtension: "json") else {
            throw WikiMQADatasetError.missingBundleFile(variant: variant)
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
    case missingBundleFile(variant: WikiMQADatasetVariant)

    var errorDescription: String? {
        switch self {
        case .missingBundleFile(let variant):
            return "\(variant.bundleFilename).json not found in app bundle."
        }
    }
}
