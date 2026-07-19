import Foundation

/// 33 passages and 10 queries from `Simple_test.txt` (baking, dogs, solar system).
enum QualityPassage: Int, CaseIterable, Identifiable {
    case p1 = 1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11
    case p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22
    case p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33
    case p34, p35, p36

    var id: Int { rawValue }

    var text: String {
        switch self {
        case .p1:
            return "Always preheat your kitchen oven before mixing any dry ingredients for a cake recipe."
        case .p2:
            return "White sugar melts quickly when exposed to direct heat. If you leave it on the stove top for too long without stirring it with a wooden spoon, the sugar will turn brown, burn, and ruin the flavor of your dessert."
        case .p3:
            return "Baking chocolate chip cookies requires careful attention to the temperature of your butter. If the butter is melted completely in the microwave, the cookies will spread out flat on the baking sheet and become greasy. If the butter is too cold, it will not mix well with the brown sugar. The best method is to leave the butter on the kitchen counter for two hours until it reaches room temperature. Room temperature butter traps tiny pockets of air when beaten with sugar, which helps the cookies bake into a soft, thick, and chewy texture that holds its shape perfectly."
        case .p4:
            return "Large chicken eggs should always be cracked into a separate small bowl first to check for bad shells."
        case .p5:
            return "Baking powder and baking soda are not the same ingredient. Baking powder contains its own acid and reacts when it gets wet, while baking soda needs an outside acid like lemon juice or buttermilk to create the gas bubbles that make bread rise."
        case .p6:
            return "Clean all flour spills off your kitchen counters using a dry towel, never a wet sponge."
        case .p7:
            return "Yeast needs warm water between 38 and 43 degrees Celsius to grow. If the water is colder than that, the yeast will stay asleep, and if the water is hotter, the heat will instantly kill the yeast cells and the bread will not rise."
        case .p8:
            return "Salt is a critical ingredient in sweet dough mixtures even though you cannot taste it directly. Salt strengthens the gluten frames inside the flour bread dough, which prevents the dough from tearing apart when it expands. It also slows down the yeast growth slightly so the dough does not rise too fast and collapse in the hot oven. If you forget to add the small teaspoon of salt to your bread recipe, the final loaf will turn out pale, tasteless, and have a weak structure that falls apart when sliced."
        case .p9:
            return "Store your vanilla extract in a dark cupboard away from direct sunlight exposure."
        case .p10:
            return "Heavy whipping cream must be kept very cold before you beat it. If the cream gets warm, the fat molecules will separate instead of holding air, and you will end up with homemade butter instead of fluffy cream."
        case .p11:
            return "Sifting your flour through a fine wire mesh helps remove large lumps and makes your birthday cakes much lighter."
        case .p12:
            return "Golden Retrievers are active sporting dogs that need at least one hour of outdoor exercise every single day."
        case .p13:
            return "Young puppies learn their names quickly when you use positive reinforcement techniques. Every time you say the puppy's name and they look at your face, instantly hand them a small piece of chicken or a tasty treat so they associate the sound with good things."
        case .p14:
            return "Crate training is an effective method for housebreaking a new dog because dogs naturally dislike sleeping in a dirty space. The crate should be just large enough for the puppy to stand up, turn around completely, and lie down comfortably. If the crate is too big, the puppy might use one corner as a bathroom and sleep in the other corner, which ruins the training process. You should never use the crate as a place for harsh punishment, or the dog will become afraid of it. Instead, feed them meals inside the crate to make it a safe home."
        case .p15:
            return "Brush your dog's thick fur coat three times a week to prevent knots and tangled mats."
        case .p16:
            return "Teaching a dog to walk on a loose leash takes patience. If the dog pulls forward tightly, stop moving immediately and stand completely still like a tree until the leash goes slack again. Only start walking forward when the dog looks back at you."
        case .p17:
            return "Never feed your dog chocolate, onions, or grapes because these specific common human foods are highly toxic."
        case .p18:
            return "Adult dogs need their nails trimmed once every month. If you hear their nails clicking loudly on the hard kitchen floor when they walk, it means the nails are getting too long and could cause paw pain or joint problems if left unclipped."
        case .p19:
            return "Golden Retrievers are famous for their love of swimming because their inner undercoat is water-resistant. This special double coat keeps their skin completely dry and warm even when they are paddling around in cold lake water during winter months. However, because their ears drop down flat against their head, water easily gets trapped inside the dark ear canal after a swim. This trapped moisture creates a perfect home for bacteria to grow. Owners must always dry the inside of the dog's ears thoroughly with a soft cotton cloth after every single swimming session."
        case .p20:
            return "Keep fresh, clean drinking water available in a heavy ceramic bowl at all times."
        case .p21:
            return "Socializing a young puppy means exposing them to new sounds, strange places, and friendly people before they turn four months old. This early exposure helps them grow into a calm, confident adult dog who is not afraid of the world."
        case .p22:
            return "Checking your dog's skin for small crawling ticks after walks in tall grass prevents dangerous illnesses."
        case .p23:
            return "The Sun is a massive star that holds ninety-nine percent of the solar system's total mass."
        case .p24:
            return "Mercury is the closest planet to the Sun, but it is actually not the hottest planet. Because Mercury has no thick atmosphere to trap heat, its dark side gets incredibly cold, dropping down to minus one hundred and seventy degrees Celsius at night."
        case .p25:
            return "Venus is the brightest natural object in our night sky after the Moon because it is covered in thick clouds. These heavy clouds are made of sulfuric acid and act like a giant mirror, reflecting most of the sunlight back out into deep space. However, these same clouds also create a powerful greenhouse effect on the planet's surface. The thick atmosphere traps the Sun's heat like a heavy blanket, raising the surface temperature to a constant four hundred and sixty degrees Celsius. This extreme heat is hot enough to melt solid lead, making Venus the hottest planet in the solar system, even though Mercury sits much closer to the Sun."
        case .p26:
            return "Mars is called the Red Planet because its surface dirt is covered in rusty iron oxide dust."
        case .p27:
            return "Jupiter is the largest planet in our solar system and spins faster than any other world. One single day on Jupiter lasts only ten hours, which causes strong, violent winds that create massive, colorful storm bands across its outer atmosphere that are visible through small telescopes."
        case .p28:
            return "Saturn's famous wide rings are made of billions of pieces of ice and rock."
        case .p29:
            return "Uranus is a unique blue gas giant because it spins completely on its side like a rolling ball. Scientists believe a massive object crashed into the planet long ago, tilting its rotation axis permanently during the early formation of the solar system."
        case .p30:
            return "Neptune is the most distant planet from the Sun and experiences the fastest winds in the solar system. These freezing winds can reach speeds of over two thousand kilometers per hour, which is twice as fast as a commercial jet airplane. Because it sits so far out in the cold darkness, Neptune takes nearly one hundred and sixty-five Earth years to complete just one single trip around the Sun. This means a single season on Neptune lasts for more than forty Earth years, leaving its poles in darkness for decades at a time."
        case .p31:
            return "Pluto was reclassified from a main planet to a dwarf planet in 2006."
        case .p32:
            return "The Asteroid Belt is a giant ring of rocks floating between the orbits of Mars and Jupiter. It contains millions of leftover pieces from the early days when our planets were first forming billions of years ago."
        case .p33:
            return "Comets are dirty snowballs made of frozen ice and dust that grow long tails when nearing the Sun."
        case .p34:
            return "Always rinse your paper coffee filter with hot water before adding any ground coffee beans."
        case .p35:
            return "The temperature of your brewing water is crucial for extracting the best flavor. If the water is boiling, it will scorch the grounds and make the coffee taste bitter, so let it sit for one minute off the heat before pouring."
        case .p36:
            return "A burr grinder is significantly better than a blade grinder for preparing coffee beans. Blade grinders chop the beans unevenly, resulting in a mix of fine dust and large chunks. This uneven mixture causes unpredictable extraction, making your coffee taste simultaneously sour and bitter. A burr grinder crushes the beans between two abrasive surfaces, creating uniform particles. Consistent grind size ensures that the water extracts flavor compounds at the exact same rate from every single coffee particle, leading to a smooth, balanced, and sweet cup of coffee every morning."
        }
    }
}

struct QualityTestQuery: Identifiable, Hashable {
    let id: Int
    let title: String
    let passages: [QualityPassage]
    let question: String
    let expectedAnswer: String
    let keyPhrases: [String]
    /// Override suite default max answer tokens when set.
    let maxGenTokens: Int32?

    init(
        id: Int,
        title: String,
        passages: [QualityPassage],
        question: String,
        expectedAnswer: String,
        keyPhrases: [String],
        maxGenTokens: Int32? = nil
    ) {
        self.id = id
        self.title = title
        self.passages = passages
        self.question = question
        self.expectedAnswer = expectedAnswer
        self.keyPhrases = keyPhrases
        self.maxGenTokens = maxGenTokens
    }
}

enum QualityTestSuiteKind: String, CaseIterable, Identifiable {
    case simple = "Simple_test (10 queries)"
    case reuseMax = "Reuse max (8 queries)"
    case stitchProfile = "Stitch profile (Q1, Q2)"
    case q2WarmReuse = "Warm reuse (Q2 × 2)"
    case ensureProfile = "Ensure profile (Q1,2,4,5,11,12)"
    case harder = "Harder_test (3 mega prompts)"

    var id: String { rawValue }

    var logTitle: String {
        switch self {
        case .simple: return "10-Query Quality Matrix (Simple_test)"
        case .reuseMax: return "Reuse-Max Benchmark (Simple_test Q1,2,4,5,6,7,11,12)"
        case .stitchProfile: return "Stitch Profile (Simple_test Q1, Q2 — baseline vs PCB)"
        case .q2WarmReuse: return "Warm Reuse Benchmark (Simple_test Q2 × 2)"
        case .ensureProfile: return "Ensure Profile (Simple_test Q1,2,4,5,11,12 — baseline vs PCB)"
        case .harder: return "3-Query Quality Matrix (Harder_test)"
        }
    }

    var defaultMaxGenTokens: Int32 {
        switch self {
        case .simple, .reuseMax, .stitchProfile, .q2WarmReuse, .ensureProfile: return 200
        case .harder: return 384
        }
    }

    /// When true, run all baseline queries then clear KV and run all PCB (maximizes cross-query reuse).
    var sequentialPathBenchmark: Bool {
        self == .reuseMax || self == .stitchProfile || self == .q2WarmReuse || self == .ensureProfile
    }

    /// Timing-only benchmark; skip pass/fail scoring emphasis.
    var timingOnlyBenchmark: Bool {
        self == .stitchProfile || self == .q2WarmReuse || self == .ensureProfile
    }
}

enum QualityTestSuite {
    static let passThreshold: Double = 0.70

    /// Backward-compatible alias for the Simple_test suite.
    static let queries = simpleQueries

    static func queries(for kind: QualityTestSuiteKind) -> [QualityTestQuery] {
        switch kind {
        case .simple: return simpleQueries
        case .reuseMax: return reuseMaxQueries
        case .stitchProfile: return stitchProfileQueries
        case .q2WarmReuse: return q2WarmReuseQueries
        case .ensureProfile: return ensureProfileQueries
        case .harder: return harderQueries
        }
    }

    /// Simple_test queries ordered for maximum chunk reuse (from Simple_test.txt).
    static let reuseMaxQueryIds = [1, 2, 4, 5, 6, 7, 11, 12]

    static let stitchProfileQueryIds = [1, 2]

    /// Ensure-phase profile: overlapping Simple_test queries (cold → warm reuse builds).
    static let ensureProfileQueryIds = [1, 2, 4, 5, 11, 12]

    static var stitchProfileQueries: [QualityTestQuery] {
        let byId = Dictionary(uniqueKeysWithValues: simpleQueries.map { ($0.id, $0) })
        return stitchProfileQueryIds.compactMap { byId[$0] }
    }

    static var ensureProfileQueries: [QualityTestQuery] {
        let byId = Dictionary(uniqueKeysWithValues: simpleQueries.map { ($0.id, $0) })
        return ensureProfileQueryIds.compactMap { byId[$0] }
    }

    /// Q2 twice (ids 201/202) — second PCB run should be all-cache HIT after warm-up.
    static var q2WarmReuseQueries: [QualityTestQuery] {
        guard let q2 = simpleQueries.first(where: { $0.id == 2 }) else { return [] }
        return [
            QualityTestQuery(
                id: 201,
                title: "Sugar & leavening (run 1)",
                passages: q2.passages,
                question: q2.question,
                expectedAnswer: q2.expectedAnswer,
                keyPhrases: q2.keyPhrases,
                maxGenTokens: q2.maxGenTokens
            ),
            QualityTestQuery(
                id: 202,
                title: "Sugar & leavening (run 2 — warm)",
                passages: q2.passages,
                question: q2.question,
                expectedAnswer: q2.expectedAnswer,
                keyPhrases: q2.keyPhrases,
                maxGenTokens: q2.maxGenTokens
            ),
        ]
    }

    static var reuseMaxQueries: [QualityTestQuery] {
        let byId = Dictionary(uniqueKeysWithValues: simpleQueries.map { ($0.id, $0) })
        return reuseMaxQueryIds.compactMap { byId[$0] }
    }

    static func maxGenTokens(for query: QualityTestQuery, suite: QualityTestSuiteKind) -> Int32 {
        query.maxGenTokens ?? suite.defaultMaxGenTokens
    }

    static let simpleQueries: [QualityTestQuery] = [
        QualityTestQuery(
            id: 1,
            title: "Butter & cookies",
            passages: [.p1, .p2, .p3],
            question: "Explain how the temperature of butter affects the final texture of chocolate chip cookies.",
            expectedAnswer: "Butter temperature changes cookie texture completely. If the butter is completely melted, the cookies spread out flat and become greasy. If the butter is too cold, it will not mix well with the sugar. Leaving it out to reach room temperature is best because room-temperature butter traps tiny pockets of air when mixed with sugar, making the cookies soft, thick, and chewy.",
            keyPhrases: [
                "melted", "flat", "greasy", "too cold", "room temperature",
                "chewy", "soft", "thick"
            ]
        ),
        QualityTestQuery(
            id: 2,
            title: "Sugar & leavening",
            passages: [.p1, .p2, .p3, .p4, .p5],
            question: "What happens to sugar left on a hot stove, and what is the difference between baking powder and baking soda?",
            expectedAnswer: "White sugar will melt quickly on a hot stove, but if left too long without stirring, it will turn brown and burn. For the raising agents, baking powder contains its own acid and activates when wet, whereas baking soda requires an outside acid (like lemon juice or buttermilk) to create the bubbles that make bread rise.",
            keyPhrases: [
                "brown", "burn", "baking powder", "baking soda",
                "acid", "lemon", "buttermilk", "rise"
            ]
        ),
        QualityTestQuery(
            id: 3,
            title: "Cross-domain facts",
            passages: [.p6, .p12, .p23],
            question: "Briefly list the rule for cleaning up flour, the exercise needs of a Golden Retriever, and the mass of the Sun.",
            expectedAnswer: "Flour spills should always be cleaned with a dry towel rather than a wet sponge. A Golden Retriever is an active sporting dog that requires at least one hour of outdoor exercise every day. The Sun is a massive star holding 99% of the solar system's total mass.",
            keyPhrases: [
                "dry towel", "wet sponge", "one hour", "exercise",
                "ninety-nine", "99%", "mass"
            ]
        ),
        QualityTestQuery(
            id: 4,
            title: "Crate & swimming",
            passages: [.p13, .p14, .p19],
            question: "Explain how to use a crate for training a puppy, and how to protect a Golden Retriever's ears after they go swimming.",
            expectedAnswer: "Crate training works because puppies naturally avoid sleeping where they go to the bathroom. The crate must be small enough so they cannot use one corner as a bathroom, and it should never be used for punishment. For swimming, a Golden Retriever's water-resistant undercoat keeps their skin dry, but their floppy ears trap water inside the dark ear canal. Owners must dry the inside of the ears thoroughly with a soft cloth after a swim to stop bacteria from growing.",
            keyPhrases: [
                "crate", "bathroom", "punishment", "ears", "swim",
                "dry", "bacteria", "water-resistant"
            ]
        ),
        QualityTestQuery(
            id: 5,
            title: "Name & leash",
            passages: [.p13, .p14, .p16],
            question: "How do you teach a puppy its name, and what should you do if a dog pulls tightly on its leash during a walk?",
            expectedAnswer: "You teach a puppy its name using positive reinforcement by giving it a treat or a piece of chicken immediately when it looks at your face after you say its name. If a dog pulls tightly on its leash during a walk, you should stop moving instantly and stand still like a tree until the leash goes slack, only moving forward again once the dog looks back at you.",
            keyPhrases: [
                "positive reinforcement", "treat", "chicken", "stop",
                "stand still", "tree", "leash", "slack"
            ]
        ),
        QualityTestQuery(
            id: 6,
            title: "Planets",
            passages: [.p24, .p26, .p30],
            question: "Why does Mercury get cold at night, what makes Mars red, and how fast are the winds on Neptune?",
            expectedAnswer: "Mercury gets cold at night (dropping to minus 170°C) because it lacks a thick atmosphere to trap the Sun's heat. Mars looks red because its surface dirt is covered in rusty iron oxide dust. Neptune experiences the fastest winds in the solar system, screaming at speeds over 2,000 kilometers per hour.",
            keyPhrases: [
                "atmosphere", "cold", "iron oxide", "red",
                "neptune", "winds", "2000", "2,000", "fastest"
            ]
        ),
        QualityTestQuery(
            id: 7,
            title: "Venus & Uranus",
            passages: [.p25, .p26, .p29],
            question: "Why is Venus hotter than Mercury, and why does Uranus spin on its side?",
            expectedAnswer: "Even though Mercury is closer to the sun, Venus is hotter (reaching a constant 460°C) because its thick sulfuric acid clouds trap heat in a massive greenhouse effect. Uranus spins completely on its side because scientists believe a huge object slammed into the planet long ago during the early days of the solar system.",
            keyPhrases: [
                "venus", "hotter", "greenhouse", "sulfuric",
                "uranus", "side", "crashed", "tilt"
            ]
        ),
        QualityTestQuery(
            id: 8,
            title: "Jupiter & asteroids",
            passages: [.p27, .p32],
            question: "How long is a day on Jupiter, what causes its violent winds, and what is the Asteroid Belt made out of?",
            expectedAnswer: "One single day on Jupiter lasts only ten hours. This incredibly fast rotation causes powerful, violent winds that create massive storm bands in its outer atmosphere. The Asteroid Belt is a giant floating ring made of millions of leftover rocks from the early days of planet formation billions of years ago.",
            keyPhrases: [
                "ten hours", "10 hours", "winds", "rotation",
                "asteroid belt", "rocks", "mars", "jupiter"
            ]
        ),
        QualityTestQuery(
            id: 9,
            title: "Yeast, salt, comets",
            passages: [.p7, .p8, .p33],
            question: "What temperature does yeast need to stay alive, why do we add salt to sweet dough, and what happens to a comet near the Sun?",
            expectedAnswer: "Yeast needs warm water between 38°C and 43°C; colder water leaves it asleep and hotter water kills it. Salt is added to sweet dough to strengthen the gluten frame and slow down yeast growth so the bread doesn't collapse. When a comet (which is a dirty snowball of ice and dust) gets close to the Sun, it grows a long tail.",
            keyPhrases: [
                "38", "43", "yeast", "salt", "gluten",
                "comet", "tail", "ice", "dust"
            ]
        ),
        QualityTestQuery(
            id: 10,
            title: "Storage & rings",
            passages: [.p9, .p10, .p15, .p28],
            question: "Provide quick tips on storing vanilla extract, whipping cream properties, brushing a dog, and the composition of Saturn's rings.",
            expectedAnswer: "Vanilla extract must be kept in a dark cupboard away from direct sunlight. Heavy whipping cream must be kept very cold before beating so it doesn't turn into butter. A dog's fur should be brushed three times a week to stop mats and knots. Saturn's wide rings are composed of billions of pieces of ice and rock.",
            keyPhrases: [
                "dark cupboard", "sunlight", "cold", "cream", "butter",
                "three times", "brush", "saturn", "ice", "rock"
            ]
        ),
        QualityTestQuery(
            id: 11,
            title: "Coffee grinder",
            passages: [.p34, .p35, .p36],
            question: "Explain why a burr grinder is better than a blade grinder, and what happens if you pour boiling water directly on coffee grounds.",
            expectedAnswer: "A burr grinder is superior because it crushes beans into uniform particles for balanced extraction, whereas a blade grinder chops them unevenly, resulting in sour and bitter coffee. If you pour boiling water directly on the grounds, it will scorch them and create a bitter taste; water should sit for one minute off the heat first.",
            keyPhrases: [
                "burr", "blade", "uniform", "bitter", "sour",
                "boiling", "scorch", "one minute", "rinse"
            ]
        ),
        QualityTestQuery(
            id: 12,
            title: "Filter rinse",
            passages: [.p34, .p35],
            question: "What is the very first step you should take with a paper filter, and how long should boiling water sit before brewing?",
            expectedAnswer: "The first step is to always rinse the paper coffee filter with hot water before adding your grounds. Boiling water should be left to sit for one minute off the heat before pouring.",
            keyPhrases: [
                "rinse", "filter", "hot water", "one minute", "boiling"
            ]
        ),
    ]

    static let harderQueries: [QualityTestQuery] = [
        QualityTestQuery(
            id: 1,
            title: "Kitchen masterclass",
            passages: [.p1, .p2, .p3, .p4, .p5, .p6, .p7, .p8, .p9, .p10, .p11],
            question: """
                Using all the provided kitchen rules, write a comprehensive summary of how temperature specifically affects baking ingredients. You must include the exact temperature behaviors of butter, sugar, yeast, and heavy whipping cream. Afterward, briefly list the proper physical handling procedures for flour, eggs, and salt.
                """,
            expectedAnswer: """
                Temperature dictates baking success: Butter must be room temperature to trap air and prevent flat, greasy cookies; sugar will turn brown and burn if left on high heat without stirring; yeast requires warm water exactly between 38°C and 43°C to grow without dying; and heavy whipping cream must be kept very cold or it will turn into butter. For handling: flour spills must be cleaned with a dry towel (and sifted for cakes), large eggs should be cracked into a separate bowl first to check for bad shells, and salt must be added to sweet dough to strengthen the gluten and prevent the dough from rising too fast and collapsing.
                """,
            keyPhrases: [
                "room temperature", "flat", "greasy", "brown", "burn",
                "38", "43", "yeast", "cold", "cream", "butter",
                "dry towel", "sift", "separate bowl", "cracked", "salt", "gluten"
            ],
            maxGenTokens: 384
        ),
        QualityTestQuery(
            id: 2,
            title: "Golden Retriever encyclopedia",
            passages: [.p12, .p13, .p14, .p15, .p16, .p17, .p18, .p19, .p20, .p21, .p22],
            question: """
                Based on the provided guidelines, detail a complete care guide for a Golden Retriever. You must categorize your answer into three sections: "Training Methods" (covering crate training, name recognition, and leash walking), "Physical Grooming & Health" (covering ears, coat, and nails), and "Dangers" (covering toxic foods and outdoor pests).
                """,
            expectedAnswer: """
                Training Methods: Crate training works by using a crate just large enough to stand and turn around in, making it a safe space for meals, never punishment. Teach their name via positive reinforcement by giving a treat instantly when they look at you. For leash walking, stop like a tree if they pull, and only move forward when the leash is slack and they look back.

                Physical Grooming & Health: After their daily hour of exercise (or swimming in cold lakes thanks to their double coat), you must thoroughly dry the inside of their ears with a cloth to prevent bacteria. Brush their thick coat three times a week, and trim their nails monthly (before they click on the floor).

                Dangers: Never feed them toxic human foods like chocolate, onions, or grapes. After walks in tall grass, always check their skin for ticks to prevent dangerous illnesses.
                """,
            keyPhrases: [
                "crate", "punishment", "positive reinforcement", "treat",
                "tree", "slack", "leash", "ears", "dry", "bacteria",
                "three times", "month", "brush", "chocolate", "onions",
                "grapes", "ticks", "one hour"
            ],
            maxGenTokens: 384
        ),
        QualityTestQuery(
            id: 3,
            title: "Cross-domain stress (33 chunks)",
            passages: QualityPassage.allCases,
            question: """
                You are scanning a massive database of random facts. I need you to find and list every single specific measurement of time (e.g., hours, months, years) mentioned across all the documents, regardless of whether it is about baking, dogs, or planets. Second, identify any mention of "ice" across all documents.
                """,
            expectedAnswer: """
                Time Measurements: Two hours (leaving butter on the counter); one hour (daily exercise for a Golden Retriever); three times a week (brushing a dog's coat); once every month (trimming a dog's nails); four months old (age to socialize a puppy by); ten hours (one day on Jupiter); 165 Earth years (Neptune's trip around the sun); 40 Earth years (one season on Neptune); decades (Neptune's poles left in darkness); 2006 (year Pluto was reclassified); billions of years ago (early days of planet formation).

                Mentions of Ice: Saturn's wide rings are made of billions of pieces of ice and rock. Comets are dirty snowballs made of frozen ice and dust.
                """,
            keyPhrases: [
                "two hours", "one hour", "three times", "month",
                "four months", "ten hours", "165", "sixty-five",
                "forty", "40", "decades", "2006", "billions",
                "ice", "saturn", "comet"
            ],
            maxGenTokens: 512
        ),
    ]

    static func passageTexts(for query: QualityTestQuery) -> [String] {
        query.passages.map(\.text)
    }
}

enum QualityScorer {
    struct Result {
        let matched: [String]
        let missed: [String]
        let score: Double
        var passed: Bool { score >= QualityTestSuite.passThreshold }
    }

    static func score(answer: String, keyPhrases: [String]) -> Result {
        let normalized = normalize(answer)
        var matched: [String] = []
        var missed: [String] = []

        for phrase in keyPhrases {
            if phraseMatches(phrase, in: normalized) {
                matched.append(phrase)
            } else {
                missed.append(phrase)
            }
        }

        let score = keyPhrases.isEmpty
            ? 0
            : Double(matched.count) / Double(keyPhrases.count)
        return Result(matched: matched, missed: missed, score: score)
    }

    private static func normalize(_ text: String) -> String {
        text.lowercased()
            .replacingOccurrences(of: "°", with: "")
            .replacingOccurrences(of: "–", with: "-")
            .replacingOccurrences(of: "—", with: "-")
    }

    /// A key phrase matches if any slash-separated alternative appears.
    private static func phraseMatches(_ phrase: String, in normalizedAnswer: String) -> Bool {
        let alternatives = phrase.lowercased()
            .split(separator: "/")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        if alternatives.isEmpty {
            return false
        }

        return alternatives.contains { alt in
            normalizedAnswer.contains(alt)
        }
    }
}

enum RagInferencePath: String {
    case standardLlama = "Standard llama (full prefill)"
    case phoneCacheBlend = "PhoneCacheBlend (cache+stitch+fuse)"
    /// Modular KV reuse only: ensure + stitch, no CacheBlend fuse (recomp_ratio=0 semantics).
    case phoneCacheBlendNoFuse = "PhoneCacheBlend no-fuse (cache+stitch, ratio=0)"

    /// Uses chunk / prefix / label caches and stitch (PCB family).
    var usesChunkCache: Bool {
        switch self {
        case .phoneCacheBlend, .phoneCacheBlendNoFuse: return true
        case .standardLlama: return false
        }
    }

    /// Runs GRAPH/TOKEN CacheBlend fuse after stitch.
    var runsCacheBlendFuse: Bool {
        self == .phoneCacheBlend
    }

    var shortTag: String {
        switch self {
        case .standardLlama: return "baseline"
        case .phoneCacheBlend: return "pcb"
        case .phoneCacheBlendNoFuse: return "pcb_nofuse"
        }
    }
}

struct RagQueryResult {
    let answer: String
    let inferenceMode: String
    let e2eTtftMs: Double
    let promptTokens: Int
    /// Baseline: full prefill ms. PCB: unused (see phase fields).
    let prefillMs: Double
    /// PCB: ensure (HIT/SAVE or collect). Baseline: nil.
    let cacheEnsureMs: Double?
    /// PCB: KV concat + label/question GPU prefill (no fuse).
    let stitchMs: Double?
    /// PCB: HKVD index pick + partial recompute (GRAPH fuse).
    let fuseMs: Double?
    /// Gap from end of fuse (or stitch if no fuse) to first answer token sampled.
    let firstTokenMs: Double?
    let cacheHits: Int?
    let cacheSaves: Int?
    /// PCB: passage-body ensure HIT count (excludes prefix/labels).
    let passageCacheHits: Int?
    let passageCacheSaves: Int?
    /// PCB: stitch loaded chunk KV from RAM hot cache vs disk.
    let stitchRamHits: Int?
    let stitchDiskLoads: Int?
    /// PCB: per-component stitch timings when collected.
    let stitchBreakdown: StitchTimingBreakdown?
    /// PCB: per-component fuse timings when collected.
    let fuseBreakdown: FuseTimingBreakdown?
    /// PCB: per-component ensure timings when collected.
    let ensureBreakdown: EnsureTimingBreakdown?
    /// Set when PCB fell back to full GPU prefill.
    let fallbackReason: String?
}

struct QualityQueryRun {
    let query: QualityTestQuery
    let path: RagInferencePath
    let answer: String
    let score: QualityScorer.Result
    let e2eTtftMs: Double
    let promptTokens: Int
    let prefillMs: Double
    let cacheEnsureMs: Double?
    let stitchMs: Double?
    let fuseMs: Double?
    let firstTokenMs: Double?
    let cacheHits: Int?
    let cacheSaves: Int?
    let stitchBreakdown: StitchTimingBreakdown?
    let fuseBreakdown: FuseTimingBreakdown?
    let ensureBreakdown: EnsureTimingBreakdown?
}
