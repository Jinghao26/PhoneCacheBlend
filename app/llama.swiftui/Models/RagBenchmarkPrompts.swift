import Foundation

/// System + question layout for a RAG benchmark run (CacheBlend server scripts use per-dataset prompts).
struct RagBenchmarkPrompts: Sendable, Equatable {
    let systemPrefix: String
    let questionPrefix: String
    let answerPrefix: String

    static let standard = RagBenchmarkPrompts(
        systemPrefix: PhoneCacheBlendConfig.systemPrefix,
        questionPrefix: PhoneCacheBlendConfig.questionPrefix,
        answerPrefix: PhoneCacheBlendConfig.answerPrefix
    )

    func buildFullPrompt(passages: [String], question: String) -> String {
        let trimmedPassages = passages
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let body = trimmedPassages.enumerated().map { index, passage in
            "[\(index + 1)] \(passage)"
        }.joined(separator: "\n\n")

        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        return systemPrefix + body + questionPrefix + q + answerPrefix
    }
}
