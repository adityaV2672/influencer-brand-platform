"""
Niche-specific vocabulary used by the synthetic caption generator.

This is deliberately hand-written rather than sampled from an LLM: it keeps the
generator deterministic, dependency-free and inspectable, and it means the
topic structure the model has to recover is documented ground truth.
"""
from __future__ import annotations

# Each niche carries: subjects, actions, objects, descriptors, hashtags, brands.
NICHE_LEXICON: dict[str, dict[str, list[str]]] = {
    "Fashion": {
        "subject": ["this outfit", "the fit", "today's look", "this silhouette", "the drape",
                    "this jacket", "the tailoring", "these boots", "the layering"],
        "action": ["styled", "paired", "thrifted", "layered", "sized down", "restyled", "tailored"],
        "object": ["linen shirt", "wide-leg trousers", "trench coat", "knit vest", "slip dress",
                   "denim jacket", "loafers", "oversized blazer", "cargo pants", "midi skirt"],
        "descriptor": ["effortless", "structured", "oversized", "minimal", "tailored", "vintage",
                       "sculptural", "understated", "relaxed"],
        "hashtags": ["ootd", "styleinspo", "fashion", "streetstyle", "capsulewardrobe", "sustainablefashion",
                     "thrifted", "outfitideas", "minimalstyle", "lookbook"],
        "brands": ["Zara", "Uniqlo", "H&M", "Mango", "COS", "Levis", "Nykaa Fashion", "Bewakoof"],
        "products": ["linen shirt", "trench coat", "denim jacket", "loafers", "tote bag"],
    },
    "Beauty": {
        "subject": ["this formula", "the finish", "my routine", "this shade", "the texture",
                    "the coverage", "this serum"],
        "action": ["swatched", "layered", "patch-tested", "reviewed", "reformulated", "blended"],
        "object": ["niacinamide serum", "cream blush", "SPF 50", "retinol", "lip oil",
                   "cushion foundation", "hyaluronic acid", "clay mask", "brow gel"],
        "descriptor": ["dewy", "buildable", "non-comedogenic", "lightweight", "pigmented",
                       "fragrance-free", "long-wearing", "sheer"],
        "hashtags": ["skincare", "makeup", "beautyreview", "skincareroutine", "cleanbeauty",
                     "grwm", "makeuptutorial", "sunscreen", "skincaretips", "beautyhacks"],
        "brands": ["Minimalist", "The Ordinary", "Maybelline", "Lakme", "Cetaphil", "Nykaa", "Dot & Key"],
        "products": ["niacinamide serum", "sunscreen", "cream blush", "retinol", "lip oil"],
    },
    "Fitness": {
        "subject": ["this split", "the progression", "my deadlift", "the warm-up", "this block",
                    "recovery week", "the form cue"],
        "action": ["trained", "progressed", "deloaded", "programmed", "logged", "scaled"],
        "object": ["push-pull-legs split", "hip hinge", "zone 2 cardio", "protein intake",
                   "mobility work", "compound lifts", "tempo squats", "rest-pause sets"],
        "descriptor": ["sustainable", "progressive", "beginner-friendly", "evidence-based",
                       "high-volume", "low-impact", "consistent"],
        "hashtags": ["fitness", "gym", "strengthtraining", "homeworkout", "fitnessjourney",
                     "progressiveoverload", "mobility", "fitnesstips", "workoutroutine", "nutrition"],
        "brands": ["MyProtein", "Decathlon", "Cult Fit", "Nike", "Adidas", "HealthKart"],
        "products": ["whey protein", "resistance bands", "running shoes", "creatine", "yoga mat"],
    },
    "Food": {
        "subject": ["this recipe", "the crumb", "the sear", "this dal", "the sauce",
                    "the fermentation", "this bake"],
        "action": ["tested", "adapted", "fermented", "braised", "reduced", "proofed", "seasoned"],
        "object": ["one-pot pasta", "sourdough loaf", "paneer curry", "cold brew", "khichdi",
                   "sheet-pan dinner", "miso broth", "chutney", "biryani"],
        "descriptor": ["weeknight", "budget-friendly", "make-ahead", "high-protein", "under 30 minutes",
                       "batch-cookable", "freezer-friendly"],
        "hashtags": ["food", "recipe", "homecooking", "foodie", "easyrecipes", "mealprep",
                     "indianfood", "baking", "vegetarian", "foodphotography"],
        "brands": ["Swiggy", "Zomato", "Blue Tokai", "Amul", "Licious", "Country Delight"],
        "products": ["cold brew", "cast iron pan", "olive oil", "spice box", "air fryer"],
    },
    "Travel": {
        "subject": ["this stretch of coast", "the old quarter", "this trail", "the ferry",
                    "the guesthouse", "this valley", "the night train"],
        "action": ["hiked", "booked", "detoured", "backpacked", "wandered", "camped"],
        "object": ["shoulder season", "budget itinerary", "homestay", "sleeper bus", "trek route",
                   "visa run", "hill station", "coastal drive"],
        "descriptor": ["off-season", "walkable", "underrated", "well-connected", "affordable",
                       "quiet", "worth the detour"],
        "hashtags": ["travel", "wanderlust", "travelphotography", "backpacking", "solotravel",
                     "traveltips", "budgettravel", "hiddengems", "roadtrip", "explore"],
        "brands": ["MakeMyTrip", "Airbnb", "IRCTC", "Booking.com", "Zostel", "Skyscanner"],
        "products": ["backpack", "travel insurance", "eSIM", "hiking boots", "power bank"],
    },
    "Technology": {
        "subject": ["the battery", "this chipset", "the thermals", "the display", "this build",
                    "the software", "the port selection"],
        "action": ["benchmarked", "stress-tested", "reviewed", "teardown-ed", "compared", "flashed"],
        "object": ["mid-range phone", "mechanical keyboard", "NVMe drive", "ultrabook",
                   "wireless earbuds", "GPU", "monitor", "NAS setup"],
        "descriptor": ["overpriced", "well-built", "surprisingly good", "throttled", "future-proof",
                       "poorly optimised", "solid value"],
        "hashtags": ["tech", "gadgets", "techreview", "smartphone", "unboxing", "pcbuild",
                     "technology", "laptop", "techtips", "audio"],
        "brands": ["Samsung", "OnePlus", "Realme", "Asus", "Logitech", "boAt", "Noise"],
        "products": ["wireless earbuds", "mechanical keyboard", "smartwatch", "laptop", "power bank"],
    },
    "Gaming": {
        "subject": ["this patch", "the frame pacing", "the grind", "this build", "the meta",
                    "the boss fight", "matchmaking"],
        "action": ["cleared", "grinded", "speedran", "theorycrafted", "reviewed", "streamed"],
        "object": ["ranked ladder", "loot pool", "damage rotation", "co-op campaign", "controller settings",
                   "early access build", "battle pass"],
        "descriptor": ["balanced", "grindy", "overtuned", "polished", "buggy", "generous", "punishing"],
        "hashtags": ["gaming", "gamer", "gameplay", "twitch", "esports", "pcgaming",
                     "gamereview", "streaming", "indiegames", "letsplay"],
        "brands": ["Steam", "Xbox", "PlayStation", "Razer", "Nvidia", "Krafton"],
        "products": ["gaming mouse", "headset", "controller", "graphics card", "game key"],
    },
    "Finance": {
        "subject": ["this allocation", "the expense ratio", "the drawdown", "this SIP",
                    "the tax treatment", "the yield"],
        "action": ["rebalanced", "back-tested", "compared", "modelled", "reviewed", "laddered"],
        "object": ["index fund", "emergency fund", "debt fund", "tax-saving instrument",
                   "asset allocation", "credit card churn", "REIT", "term insurance"],
        "descriptor": ["low-cost", "tax-efficient", "risk-adjusted", "long-horizon", "boring",
                       "misunderstood", "over-marketed"],
        "hashtags": ["personalfinance", "investing", "mutualfunds", "stockmarket", "financialfreedom",
                     "moneytips", "sip", "taxplanning", "savings", "wealth"],
        "brands": ["Zerodha", "Groww", "Kuvera", "HDFC", "ICICI Direct", "Paytm Money"],
        "products": ["index fund", "term insurance", "credit card", "demat account", "health insurance"],
    },
    "Parenting": {
        "subject": ["this routine", "the transition", "bedtime", "the tantrum", "this stage",
                    "the handover", "screen time"],
        "action": ["adjusted", "survived", "reframed", "scheduled", "negotiated", "modelled"],
        "object": ["nap schedule", "weaning", "sensory play", "school run", "sleep regression",
                   "boundary setting", "toddler meals"],
        "descriptor": ["gentle", "realistic", "low-pressure", "age-appropriate", "messy",
                       "flexible", "hard-won"],
        "hashtags": ["parenting", "momlife", "toddler", "gentleparenting", "newborn",
                     "parentingtips", "kidsactivities", "dadlife", "motherhood", "familylife"],
        "brands": ["FirstCry", "Pampers", "Mamaearth", "Chicco", "Himalaya Baby"],
        "products": ["diapers", "baby carrier", "stroller", "baby lotion", "high chair"],
    },
    "Automotive": {
        "subject": ["the ride quality", "this gearbox", "the NVH", "the turbo lag",
                    "the suspension", "boot space", "the service cost"],
        "action": ["drove", "reviewed", "compared", "modified", "serviced", "track-tested"],
        "object": ["compact SUV", "hot hatch", "EV crossover", "hybrid sedan", "adventure bike",
                   "manual gearbox", "tyre upgrade"],
        "descriptor": ["planted", "underpowered", "well-damped", "thirsty", "practical",
                       "surprisingly quick", "overpriced"],
        "hashtags": ["cars", "automotive", "carreview", "ev", "bikelife", "carsofinstagram",
                     "motorsport", "roadtest", "carphotography", "autoenthusiast"],
        "brands": ["Tata Motors", "Mahindra", "Hyundai", "Royal Enfield", "Kia", "MG"],
        "products": ["dash cam", "car care kit", "tyres", "helmet", "roof rack"],
    },
    "Home & Decor": {
        "subject": ["this corner", "the lighting", "the layout", "this shelf", "the palette",
                    "storage", "the rental fix"],
        "action": ["restyled", "decluttered", "renovated", "sourced", "repainted", "organised"],
        "object": ["rental-friendly setup", "modular storage", "reading nook", "warm lighting",
                   "gallery wall", "small-space layout", "plant corner"],
        "descriptor": ["renter-friendly", "budget", "warm", "functional", "low-maintenance",
                       "clutter-free", "cosy"],
        "hashtags": ["homedecor", "interiordesign", "smallspaces", "homeorganization", "diyhome",
                     "rentaldecor", "plantparent", "homeinspo", "declutter", "cozyhome"],
        "brands": ["IKEA", "Pepperfry", "Urban Ladder", "Home Centre", "Wakefit"],
        "products": ["floor lamp", "storage bins", "mattress", "rug", "shelving unit"],
    },
    "Education": {
        "subject": ["this concept", "the syllabus", "this proof", "the revision plan",
                    "the mock test", "note-taking", "the exam pattern"],
        "action": ["broke down", "revised", "explained", "solved", "summarised", "mapped"],
        "object": ["spaced repetition", "past papers", "concept map", "study schedule",
                   "active recall", "problem sets", "exam strategy"],
        "descriptor": ["clear", "step-by-step", "exam-focused", "beginner-friendly", "concise",
                       "practical", "often-misunderstood"],
        "hashtags": ["studygram", "education", "studytips", "examprep", "learning",
                     "studymotivation", "notes", "students", "onlinelearning", "studyhacks"],
        "brands": ["Byjus", "Unacademy", "Physics Wallah", "Coursera", "Khan Academy"],
        "products": ["online course", "notebook", "tablet", "study planner", "reference book"],
    },
}

# Cross-niche connective tissue -------------------------------------------------
GENERIC_HASHTAGS = [
    "reels", "explore", "viral", "trending", "instadaily", "contentcreator",
    "creator", "collab", "review", "honest",
]

CTA_PHRASES = [
    "Link in bio.", "Comment below if you want the full list.", "Save this for later.",
    "Share this with someone who needs it.", "Follow for more.", "Swipe for the breakdown.",
    "Drop a question and I'll answer.", "Tap the link to get yours.",
]

PROMO_PHRASES = [
    "Use my code {code} for {pct}% off.", "Gifted by {brand} but all opinions are mine.",
    "Paid partnership with {brand}.", "Thanks {brand} for sponsoring this one.",
    "Limited-time offer, ends this week.", "Exclusive discount for my followers.",
    "#ad", "#sponsored",
]

QUESTION_PHRASES = [
    "What would you pick?", "Anyone else?", "Thoughts?", "Which one is your go-to?",
    "Am I the only one?", "Would you try this?",
]

# Sarcasm / irony templates. These are the *generated* ironic captions; genuine
# irony evaluation is done separately on the real labelled TweetEval-irony and
# news-headlines corpora - see src/benchmark/.
SARCASM_TEMPLATES = [
    "Oh {great}, another {object} that {fails}. Exactly what nobody asked for.",
    "Because clearly what {subject} needed was {absurd}. Brilliant.",
    "Nothing says {descriptor} like {absurd}. Truly {great} work.",
    "Sure, {absurd}. That'll definitely fix {subject}.",
    "Love how {subject} {fails} and we're all just supposed to pretend that's fine.",
    "Wow, {absurd}. Never seen that before. Groundbreaking.",
]
SARCASM_GREAT = ["great", "fantastic", "wonderful", "perfect", "amazing"]
SARCASM_FAILS = ["breaks on day two", "costs twice as much", "does absolutely nothing",
                 "ships without the basics", "solves a problem nobody had"]
SARCASM_ABSURD = ["a subscription fee", "three more steps", "another dongle",
                  "a 40% price hike", "yet another app", "a mandatory account"]

POSITIVE_OPENERS = [
    "Genuinely impressed by", "Really loving", "Cannot recommend", "So happy with",
    "Absolutely worth it -", "This exceeded expectations:",
]
NEGATIVE_OPENERS = [
    "Disappointed by", "Would not repeat", "Skip this one -", "Frustrating experience with",
    "Not worth the money:", "Regret buying",
]
NEUTRAL_OPENERS = [
    "Quick note on", "Breaking down", "Here's what I found about", "A short update on",
    "Documenting", "Some thoughts on",
]
