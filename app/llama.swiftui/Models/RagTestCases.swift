import Foundation

/// Built-in RAG prompts for reproducible on-device benchmarks.
/// Passages are composed from `RagTestChunk` so overlapping presets share identical
/// chunk text → same chunk_id on disk → HIT when running different test cases in sequence.
enum RagTestPreset: String, CaseIterable, Identifiable {
    case swapOrder2 = "Swap validate (AB+CDE)"
    case simple2 = "Simple (2)"
    case compact3 = "Compact (3)"
    case medium4 = "Medium (4)"
    case overlap5 = "Overlap (5)"
    case scaling6 = "Scaling (6)"
    case wide8 = "Wide (8)"

    var id: String { rawValue }

    var data: RagTestCaseData {
        switch self {
        case .swapOrder2:
            return RagTestCaseData(
                name: "Swap validate (AB + CDE)",
                description: """
                Four prompts validating content-addressed cache across slot reordering.
                Q1–Q2: A,B then B,A (2 chunks). Q3–Q4: C,D,E then E,C,D (3 chunks).
                Content-hash: Q2 → 2/2 HIT; Q4 → 3/3 HIT. Index-hash: Q2 → 0/2; Q4 → 0/3.
                """,
                chunkCount: 3,
                passages1: RagTestChunk.passages([.nabbesShort, .punicaShort]),
                question1: RagTestChunk.educationQuestion,
                passages2: RagTestChunk.passages([.punicaShort, .nabbesShort]),
                question2: RagTestChunk.educationQuestion,
                usePairMode: true,
                validationHint: """
                --- Swap-order validation (content-addressed cache) ---
                A = Nabbes (short), B = Punica (short)
                C = Red Dragon (short), D = Scipio bio, E = Treaty of Zama
                Q1: [1]A [2]B     → 2× SAVE
                Q2: [1]B [2]A     → 2× HIT (ids unchanged, slots swapped)
                Q3: [1]C [2]D [3]E → 3× SAVE
                Q4: [1]E [2]C [3]D → 3× HIT (rotation; index-hash would 0/3 HIT)
                """,
                sequentialQueries: [
                    RagTestSequentialQuery(
                        label: "RAG query #1 (A,B)",
                        passagesText: RagTestChunk.passages([.nabbesShort, .punicaShort]),
                        question: RagTestChunk.educationQuestion
                    ),
                    RagTestSequentialQuery(
                        label: "RAG query #2 (B,A)",
                        passagesText: RagTestChunk.passages([.punicaShort, .nabbesShort]),
                        question: RagTestChunk.educationQuestion
                    ),
                    RagTestSequentialQuery(
                        label: "RAG query #3 (C,D,E)",
                        passagesText: RagTestChunk.passages([.redDragonShort, .scipioBio, .carthageTreaty]),
                        question: RagTestChunk.warQuestion
                    ),
                    RagTestSequentialQuery(
                        label: "RAG query #4 (E,C,D)",
                        passagesText: RagTestChunk.passages([.carthageTreaty, .redDragonShort, .scipioBio]),
                        question: RagTestChunk.warQuestion
                    ),
                ]
            )
        case .simple2:
            return RagTestCaseData(
                name: "Simple (2 chunks)",
                description: "2 passages. Prompt #2 swaps chunk 2 (1/2 HIT).",
                chunkCount: 2,
                passages1: RagTestChunk.passages([.nabbesShort, .punicaShort]),
                question1: RagTestChunk.educationQuestion,
                passages2: RagTestChunk.passages([.nabbesShort, .redDragonShort]),
                question2: RagTestChunk.educationQuestion,
                usePairMode: true
            )
        case .compact3:
            return RagTestCaseData(
                name: "Compact (3 chunks)",
                description: "3 passages (~200 tok). Prompt #2 swaps chunk 3 (2/3 HIT). Shares A,B with other presets.",
                chunkCount: 3,
                passages1: RagTestChunk.passages([.nabbes, .punica, .redDragon]),
                question1: RagTestChunk.educationQuestion,
                passages2: RagTestChunk.passages([.nabbes, .punica, .scipioBio]),
                question2: RagTestChunk.educationQuestion,
                usePairMode: true
            )
        case .medium4:
            return RagTestCaseData(
                name: "Medium (4 chunks)",
                description: "4 passages (~280 tok). Prompt #2 swaps chunk 3 (3/4 HIT).",
                chunkCount: 4,
                passages1: RagTestChunk.passages([.nabbes, .punica, .hannibalTV, .hannibalNovel]),
                question1: RagTestChunk.educationQuestion,
                passages2: RagTestChunk.passages([.nabbes, .punica, .scipioBio, .hannibalNovel]),
                question2: RagTestChunk.educationQuestion,
                usePairMode: true
            )
        case .overlap5:
            return RagTestCaseData(
                name: "Overlap (5 chunks)",
                description: "5 passages (~350 tok). Prompt #2 swaps chunk 3 (4/5 HIT). Subset of Scaling.",
                chunkCount: 5,
                passages1: RagTestChunk.passages([.nabbes, .punica, .hannibalTV, .hannibalNovel, .redDragon]),
                question1: RagTestChunk.educationQuestion,
                passages2: RagTestChunk.passages([.nabbes, .punica, .scipioBio, .hannibalNovel, .redDragon]),
                question2: RagTestChunk.educationQuestion,
                usePairMode: true
            )
        case .scaling6:
            return RagTestCaseData(
                name: "Scaling (6 chunks)",
                description: "6 passages (~512 tok). Prompt #2 swaps chunk 3 (5/6 HIT). Primary benchmark preset.",
                chunkCount: 6,
                passages1: RagTestChunk.passages([.nabbes, .punica, .hannibalTV, .hannibalNovel, .redDragon, .battleCissa]),
                question1: RagTestChunk.educationQuestion,
                passages2: RagTestChunk.passages([.nabbes, .punica, .scipioBio, .hannibalNovel, .redDragon, .battleCissa]),
                question2: RagTestChunk.educationQuestion,
                usePairMode: true
            )
        case .wide8:
            return RagTestCaseData(
                name: "Wide (8 chunks)",
                description: "8 passages (~680 tok). Prompt #2 swaps chunk 3 (7/8 HIT). Stress test near n_ctx.",
                chunkCount: 8,
                passages1: RagTestChunk.passages([
                    .nabbes, .punica, .hannibalTV, .hannibalNovel,
                    .redDragon, .battleCissa, .scipioBio, .carthageTreaty
                ]),
                question1: RagTestChunk.educationQuestion,
                passages2: RagTestChunk.passages([
                    .nabbes, .punica, .hannibalLecter, .hannibalNovel,
                    .redDragon, .battleCissa, .scipioBio, .carthageTreaty
                ]),
                question2: RagTestChunk.educationQuestion,
                usePairMode: true
            )
        }
    }
}

struct RagTestSequentialQuery {
    let label: String
    let passagesText: String
    let question: String
}

struct RagTestCaseData {
    let name: String
    let description: String
    let chunkCount: Int
    let passages1: String
    let question1: String
    let passages2: String?
    let question2: String?
    let usePairMode: Bool
    /// Printed at the start of Ask x2 when set (validation / benchmark guidance).
    let validationHint: String?
    /// When set, runs these in order instead of the two-editor pair fields.
    let sequentialQueries: [RagTestSequentialQuery]?

    init(
        name: String,
        description: String,
        chunkCount: Int,
        passages1: String,
        question1: String,
        passages2: String? = nil,
        question2: String? = nil,
        usePairMode: Bool = false,
        validationHint: String? = nil,
        sequentialQueries: [RagTestSequentialQuery]? = nil
    ) {
        self.name = name
        self.description = description
        self.chunkCount = chunkCount
        self.passages1 = passages1
        self.question1 = question1
        self.passages2 = passages2
        self.question2 = question2
        self.usePairMode = usePairMode
        self.validationHint = validationHint
        self.sequentialQueries = sequentialQueries
    }
}

// MARK: - Shared passage library (identical text → same chunk_id across presets)

private enum RagTestChunk {
    static let educationQuestion = "Where was the author of Hannibal and Scipio educated?"
    static let warQuestion = "What ended the Second Punic War and who defeated Hannibal?"

    enum ID {
        case nabbesShort
        case punicaShort
        case redDragonShort
        case nabbes
        case punica
        case hannibalTV
        case hannibalNovel
        case redDragon
        case battleCissa
        case scipioBio
        case hannibalLecter
        case carthageTreaty
    }

    static func text(_ id: ID) -> String {
        switch id {
        case .nabbesShort:
            return """
            Hannibal and Scipio is a Caroline era stage play written by Thomas Nabbes. It was first performed in 1635. He was educated at Exeter College, Oxford in 1621.
            """
        case .punicaShort:
            return """
            The Punica is a Latin epic poem by Silius Italicus about the Second Punic War between Hannibal and Scipio Africanus.
            """
        case .redDragonShort:
            return """
            Red Dragon is a 2002 horror film based on the novel by Thomas Harris and stars Anthony Hopkins as Hannibal Lecter.
            """
        case .nabbes:
            return """
            Hannibal and Scipio is a Caroline era stage play, a classical tragedy written by Thomas Nabbes. The play was first performed in 1635 by Queen Henrietta's Men. Thomas Nabbes was educated at Exeter College, Oxford in 1621. He left the university without taking a degree and began a career in London as a dramatist around 1630.
            """
        case .punica:
            return """
            The Punica is a Latin epic poem in seventeen books by Silius Italicus, comprising over twelve thousand lines. Its theme is the Second Punic War and the conflict between the generals Hannibal and Scipio Africanus. The poem was rediscovered in the fifteenth century and became an important source for Renaissance writers studying Roman history.
            """
        case .hannibalTV:
            return """
            Hannibal is an American psychological horror television series developed by Bryan Fuller for NBC. The series is based on characters from Thomas Harris novels including Red Dragon and Hannibal. It focuses on the relationship between FBI investigator Will Graham and Dr. Hannibal Lecter during Graham's investigation of a serial killer in Minnesota.
            """
        case .hannibalNovel:
            return """
            Hannibal is a 1995 historical novel by Scottish writer Ross Leckie. The book relates the exploits of Hannibal's invasion of Italy beginning in 218 BC, narrated by the Carthaginian general in his retirement. It was the first novel in Leckie's Carthage trilogy covering the Punic Wars and received mixed reviews for its violent depictions.
            """
        case .redDragon:
            return """
            Red Dragon is a 2002 horror film directed by Brett Ratner and based on the Thomas Harris novel of the same title. Anthony Hopkins stars as psychiatrist and serial killer Dr. Hannibal Lecter. The film is a prequel to The Silence of the Lambs and explores the origin of the relationship between Lecter and FBI profiler Will Graham.
            """
        case .battleCissa:
            return """
            The Battle of Cissa was part of the Second Punic War, fought in 218 BC near the Greek town of Tarraco in northeastern Iberia. A Roman army under Gnaeus Cornnelius Scipio Calvus defeated a Carthaginian army under Hanno. This was among the first Roman victories against Hannibal's forces in Spain and helped secure territory north of the Ebro River.
            """
        case .scipioBio:
            return """
            Scipio Africanus was a Roman general and statesman who defeated Hannibal at the Battle of Zama in 202 BC, ending the Second Punic War. He was educated in Rome and showed early military talent during the Battle of Ticinus. His strategic innovations and political influence earned him the name Africanus and a lasting place in Roman history.
            """
        case .hannibalLecter:
            return """
            Hannibal Lecter is a fictional character created by novelist Thomas Harris. Lecter is a brilliant psychiatrist and cannibalistic serial killer who first appeared in the 1981 novel Red Dragon. The character was portrayed by Anthony Hopkins in several films and became one of the most iconic villains in modern thriller fiction.
            """
        case .carthageTreaty:
            return """
            The Treaty of Zama ended the Second Punic War in 201 BC between Rome and Carthage. Carthage surrendered its navy, paid a large indemnity, and agreed not to wage war without Roman consent. The terms severely weakened Carthage and shifted Mediterranean power toward Rome for the next century.
            """
        }
    }

    static func passages(_ chunks: [ID]) -> String {
        chunks.map { text($0) }.joined(separator: "\n\n---\n\n")
    }
}
