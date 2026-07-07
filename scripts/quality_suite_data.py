"""Canonical Simple_test / Harder_test fixtures (mirrors QualityTestCases.swift)."""

from __future__ import annotations

PASS_THRESHOLD = 0.70

SYSTEM_PREFIX = (
    "You are a helpful assistant. Answer the question using only the passages below. "
    "Be concise.\n\nPassages:\n"
)
QUESTION_PREFIX = "\n\nQuestion: "
ANSWER_PREFIX = "\nAnswer:"

PASSAGES: dict[int, str] = {
    1: "Always preheat your kitchen oven before mixing any dry ingredients for a cake recipe.",
    2: (
        "White sugar melts quickly when exposed to direct heat. If you leave it on the stove top "
        "for too long without stirring it with a wooden spoon, the sugar will turn brown, burn, "
        "and ruin the flavor of your dessert."
    ),
    3: (
        "Baking chocolate chip cookies requires careful attention to the temperature of your butter. "
        "If the butter is melted completely in the microwave, the cookies will spread out flat on the "
        "baking sheet and become greasy. If the butter is too cold, it will not mix well with the "
        "brown sugar. The best method is to leave the butter on the kitchen counter for two hours until "
        "it reaches room temperature. Room temperature butter traps tiny pockets of air when beaten with "
        "sugar, which helps the cookies bake into a soft, thick, and chewy texture that holds its shape "
        "perfectly."
    ),
    4: "Large chicken eggs should always be cracked into a separate small bowl first to check for bad shells.",
    5: (
        "Baking powder and baking soda are not the same ingredient. Baking powder contains its own acid "
        "and reacts when it gets wet, while baking soda needs an outside acid like lemon juice or "
        "buttermilk to create the gas bubbles that make bread rise."
    ),
    6: "Clean all flour spills off your kitchen counters using a dry towel, never a wet sponge.",
    7: (
        "Yeast needs warm water between 38 and 43 degrees Celsius to grow. If the water is colder than "
        "that, the yeast will stay asleep, and if the water is hotter, the heat will instantly kill the "
        "yeast cells and the bread will not rise."
    ),
    8: (
        "Salt is a critical ingredient in sweet dough mixtures even though you cannot taste it directly. "
        "Salt strengthens the gluten frames inside the flour bread dough, which prevents the dough from "
        "tearing apart when it expands. It also slows down the yeast growth slightly so the dough does not "
        "rise too fast and collapse in the hot oven. If you forget to add the small teaspoon of salt to "
        "your bread recipe, the final loaf will turn out pale, tasteless, and have a weak structure that "
        "falls apart when sliced."
    ),
    9: "Store your vanilla extract in a dark cupboard away from direct sunlight exposure.",
    10: (
        "Heavy whipping cream must be kept very cold before you beat it. If the cream gets warm, the fat "
        "molecules will separate instead of holding air, and you will end up with homemade butter instead "
        "of fluffy cream."
    ),
    11: "Sifting your flour through a fine wire mesh helps remove large lumps and makes your birthday cakes much lighter.",
    12: "Golden Retrievers are active sporting dogs that need at least one hour of outdoor exercise every single day.",
    13: (
        "Young puppies learn their names quickly when you use positive reinforcement techniques. Every time "
        "you say the puppy's name and they look at your face, instantly hand them a small piece of chicken "
        "or a tasty treat so they associate the sound with good things."
    ),
    14: (
        "Crate training is an effective method for housebreaking a new dog because dogs naturally dislike "
        "sleeping in a dirty space. The crate should be just large enough for the puppy to stand up, turn "
        "around completely, and lie down comfortably. If the crate is too big, the puppy might use one "
        "corner as a bathroom and sleep in the other corner, which ruins the training process. You should "
        "never use the crate as a place for harsh punishment, or the dog will become afraid of it. Instead, "
        "feed them meals inside the crate to make it a safe home."
    ),
    15: "Brush your dog's thick fur coat three times a week to prevent knots and tangled mats.",
    16: (
        "Teaching a dog to walk on a loose leash takes patience. If the dog pulls forward tightly, stop "
        "moving immediately and stand completely still like a tree until the leash goes slack again. Only "
        "start walking forward when the dog looks back at you."
    ),
    17: "Never feed your dog chocolate, onions, or grapes because these specific common human foods are highly toxic.",
    18: (
        "Adult dogs need their nails trimmed once every month. If you hear their nails clicking loudly on "
        "the hard kitchen floor when they walk, it means the nails are getting too long and could cause "
        "paw pain or joint problems if left unclipped."
    ),
    19: (
        "Golden Retrievers are famous for their love of swimming because their inner undercoat is "
        "water-resistant. This special double coat keeps their skin completely dry and warm even when "
        "they are paddling around in cold lake water during winter months. However, because their ears "
        "drop down flat against their head, water easily gets trapped inside the dark ear canal after a "
        "swim. This trapped moisture creates a perfect home for bacteria to grow. Owners must always dry "
        "the inside of the dog's ears thoroughly with a soft cotton cloth after every single swimming session."
    ),
    20: "Keep fresh, clean drinking water available in a heavy ceramic bowl at all times.",
    21: (
        "Socializing a young puppy means exposing them to new sounds, strange places, and friendly people "
        "before they turn four months old. This early exposure helps them grow into a calm, confident "
        "adult dog who is not afraid of the world."
    ),
    22: "Checking your dog's skin for small crawling ticks after walks in tall grass prevents dangerous illnesses.",
    23: "The Sun is a massive star that holds ninety-nine percent of the solar system's total mass.",
    24: (
        "Mercury is the closest planet to the Sun, but it is actually not the hottest planet. Because "
        "Mercury has no thick atmosphere to trap heat, its dark side gets incredibly cold, dropping down "
        "to minus one hundred and seventy degrees Celsius at night."
    ),
    25: (
        "Venus is the brightest natural object in our night sky after the Moon because it is covered in "
        "thick clouds. These heavy clouds are made of sulfuric acid and act like a giant mirror, reflecting "
        "most of the sunlight back out into deep space. However, these same clouds also create a powerful "
        "greenhouse effect on the planet's surface. The thick atmosphere traps the Sun's heat like a heavy "
        "blanket, raising the surface temperature to a constant four hundred and sixty degrees Celsius. "
        "This extreme heat is hot enough to melt solid lead, making Venus the hottest planet in the solar "
        "system, even though Mercury sits much closer to the Sun."
    ),
    26: "Mars is called the Red Planet because its surface dirt is covered in rusty iron oxide dust.",
    27: (
        "Jupiter is the largest planet in our solar system and spins faster than any other world. One "
        "single day on Jupiter lasts only ten hours, which causes strong, violent winds that create "
        "massive, colorful storm bands across its outer atmosphere that are visible through small telescopes."
    ),
    28: "Saturn's famous wide rings are made of billions of pieces of ice and rock.",
    29: (
        "Uranus is a unique blue gas giant because it spins completely on its side like a rolling ball. "
        "Scientists believe a massive object crashed into the planet long ago, tilting its rotation axis "
        "permanently during the early formation of the solar system."
    ),
    30: (
        "Neptune is the most distant planet from the Sun and experiences the fastest winds in the solar "
        "system. These freezing winds can reach speeds of over two thousand kilometers per hour, which "
        "is twice as fast as a commercial jet airplane. Because it sits so far out in the cold darkness, "
        "Neptune takes nearly one hundred and sixty-five Earth years to complete just one single trip "
        "around the Sun. This means a single season on Neptune lasts for more than forty Earth years, "
        "leaving its poles in darkness for decades at a time."
    ),
    31: "Pluto was reclassified from a main planet to a dwarf planet in 2006.",
    32: (
        "The Asteroid Belt is a giant ring of rocks floating between the orbits of Mars and Jupiter. It "
        "contains millions of leftover pieces from the early days when our planets were first forming "
        "billions of years ago."
    ),
    33: "Comets are dirty snowballs made of frozen ice and dust that grow long tails when nearing the Sun.",
}


def format_passage_chunk(index: int, passage: str) -> str:
    return f"[{index + 1}] {passage.strip()}\n\n"


def passage_texts(passage_ids: list[int]) -> list[str]:
    return [PASSAGES[i] for i in passage_ids]


SIMPLE_QUERIES = [
    {
        "id": 1,
        "title": "Butter & cookies",
        "passage_ids": [1, 2, 3],
        "question": "Explain how the temperature of butter affects the final texture of chocolate chip cookies.",
        "key_phrases": ["melted", "flat", "greasy", "too cold", "room temperature", "chewy", "soft", "thick"],
        "max_tokens": 200,
    },
    {
        "id": 2,
        "title": "Sugar & leavening",
        "passage_ids": [1, 2, 3, 4, 5],
        "question": "What happens to sugar left on a hot stove, and what is the difference between baking powder and baking soda?",
        "key_phrases": ["brown", "burn", "baking powder", "baking soda", "acid", "lemon", "buttermilk", "rise"],
        "max_tokens": 200,
    },
    {
        "id": 3,
        "title": "Cross-domain facts",
        "passage_ids": [6, 12, 23],
        "question": "Briefly list the rule for cleaning up flour, the exercise needs of a Golden Retriever, and the mass of the Sun.",
        "key_phrases": ["dry towel", "wet sponge", "one hour", "exercise", "ninety-nine", "99%", "mass"],
        "max_tokens": 200,
    },
    {
        "id": 4,
        "title": "Crate & swimming",
        "passage_ids": [13, 14, 19],
        "question": "Explain how to use a crate for training a puppy, and how to protect a Golden Retriever's ears after they go swimming.",
        "key_phrases": ["crate", "bathroom", "punishment", "ears", "swim", "dry", "bacteria", "water-resistant"],
        "max_tokens": 200,
    },
    {
        "id": 5,
        "title": "Name & leash",
        "passage_ids": [13, 14, 16],
        "question": "How do you teach a puppy its name, and what should you do if a dog pulls tightly on its leash during a walk?",
        "key_phrases": ["positive reinforcement", "treat", "chicken", "stop", "stand still", "tree", "leash", "slack"],
        "max_tokens": 200,
    },
    {
        "id": 6,
        "title": "Planets",
        "passage_ids": [24, 26, 30],
        "question": "Why does Mercury get cold at night, what makes Mars red, and how fast are the winds on Neptune?",
        "key_phrases": ["atmosphere", "cold", "iron oxide", "red", "neptune", "winds", "2000", "2,000", "fastest"],
        "max_tokens": 200,
    },
    {
        "id": 7,
        "title": "Venus & Uranus",
        "passage_ids": [25, 26, 29],
        "question": "Why is Venus hotter than Mercury, and why does Uranus spin on its side?",
        "key_phrases": ["venus", "hotter", "greenhouse", "sulfuric", "uranus", "side", "crashed", "tilt"],
        "max_tokens": 200,
    },
    {
        "id": 8,
        "title": "Jupiter & asteroids",
        "passage_ids": [27, 32],
        "question": "How long is a day on Jupiter, what causes its violent winds, and what is the Asteroid Belt made out of?",
        "key_phrases": ["ten hours", "10 hours", "winds", "rotation", "asteroid belt", "rocks", "mars", "jupiter"],
        "max_tokens": 200,
    },
    {
        "id": 9,
        "title": "Yeast, salt, comets",
        "passage_ids": [7, 8, 33],
        "question": "What temperature does yeast need to stay alive, why do we add salt to sweet dough, and what happens to a comet near the Sun?",
        "key_phrases": ["38", "43", "yeast", "salt", "gluten", "comet", "tail", "ice", "dust"],
        "max_tokens": 200,
    },
    {
        "id": 10,
        "title": "Storage & rings",
        "passage_ids": [9, 10, 15, 28],
        "question": "Provide quick tips on storing vanilla extract, whipping cream properties, brushing a dog, and the composition of Saturn's rings.",
        "key_phrases": ["dark cupboard", "sunlight", "cold", "cream", "butter", "three times", "brush", "saturn", "ice", "rock"],
        "max_tokens": 200,
    },
]

HARDER_QUERIES = [
    {
        "id": 1,
        "title": "Kitchen masterclass",
        "passage_ids": list(range(1, 12)),
        "question": (
            "Using all the provided kitchen rules, write a comprehensive summary of how temperature "
            "specifically affects baking ingredients. You must include the exact temperature behaviors "
            "of butter, sugar, yeast, and heavy whipping cream. Afterward, briefly list the proper "
            "physical handling procedures for flour, eggs, and salt."
        ),
        "key_phrases": [
            "room temperature", "flat", "greasy", "brown", "burn", "38", "43", "yeast", "cold",
            "cream", "butter", "dry towel", "sift", "separate bowl", "cracked", "salt", "gluten",
        ],
        "max_tokens": 384,
    },
    {
        "id": 2,
        "title": "Golden Retriever encyclopedia",
        "passage_ids": list(range(12, 23)),
        "question": (
            'Based on the provided guidelines, detail a complete care guide for a Golden Retriever. '
            'You must categorize your answer into three sections: "Training Methods" (covering crate '
            'training, name recognition, and leash walking), "Physical Grooming & Health" (covering '
            'ears, coat, and nails), and "Dangers" (covering toxic foods and outdoor pests).'
        ),
        "key_phrases": [
            "crate", "punishment", "positive reinforcement", "treat", "tree", "slack", "leash",
            "ears", "dry", "bacteria", "three times", "month", "brush", "chocolate", "onions",
            "grapes", "ticks", "one hour",
        ],
        "max_tokens": 384,
    },
    {
        "id": 3,
        "title": "Cross-domain stress (33 chunks)",
        "passage_ids": list(range(1, 34)),
        "question": (
            "You are scanning a massive database of random facts. I need you to find and list every "
            'single specific measurement of time (e.g., hours, months, years) mentioned across all the '
            'documents, regardless of whether it is about baking, dogs, or planets. Second, identify '
            'any mention of "ice" across all documents.'
        ),
        "key_phrases": [
            "two hours", "one hour", "three times", "month", "four months", "ten hours", "165",
            "sixty-five", "forty", "40", "decades", "2006", "billions", "ice", "saturn", "comet",
        ],
        "max_tokens": 512,
    },
]

SUITES = {
    "simple": {
        "name": "Simple_test (10 queries)",
        "default_max_tokens": 200,
        "queries": SIMPLE_QUERIES,
    },
    "harder": {
        "name": "Harder_test (3 mega prompts)",
        "default_max_tokens": 384,
        "queries": HARDER_QUERIES,
    },
}
