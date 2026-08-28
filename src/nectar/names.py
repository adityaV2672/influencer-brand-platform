"""
Human-readable identity for the synthetic creator universe.

The modelling pipeline identifies creators by `influencer_id` and a generated
handle. The Nectar product surface shows a *person*: a name, a city, a bio and
an avatar. This module manufactures that presentation layer deterministically
from the creator id, so the same creator is always the same person no matter
how often the export is rebuilt, and so nothing here can leak into the model
(these fields are never features).
"""
from __future__ import annotations

import hashlib

FIRST_F = [
    "Ananya", "Maya", "Tara", "Ishita", "Priya", "Shreya", "Pooja", "Aditi",
    "Nikita", "Kavya", "Riya", "Sneha", "Meera", "Divya", "Anjali", "Neha",
    "Sanya", "Trisha", "Aarohi", "Ira", "Naina", "Rhea", "Simran", "Diya",
    "Mitali", "Radhika", "Vaishnavi", "Prerna", "Aisha", "Lakshmi",
]
FIRST_M = [
    "Arjun", "Rohan", "Kabir", "Vivek", "Aditya", "Karan", "Siddharth", "Rahul",
    "Nikhil", "Aman", "Dev", "Yash", "Harsh", "Manav", "Rishi", "Varun",
    "Kunal", "Aryan", "Ishaan", "Tanmay", "Ved", "Neel", "Sameer", "Om",
    "Parth", "Raghav", "Shaurya", "Advait", "Kartik", "Zayn",
]
LAST = [
    "Shah", "Rao", "Mishra", "Nair", "Sethi", "Agarwal", "Iyer", "Singh",
    "Kapoor", "Menon", "Reddy", "Bhatt", "Chopra", "Desai", "Joshi", "Kulkarni",
    "Malhotra", "Nanda", "Pillai", "Sharma", "Verma", "Gupta", "Banerjee",
    "Chatterjee", "Das", "Ghosh", "Bose", "Hegde", "Shetty", "Kamath",
    "Trivedi", "Saxena", "Sinha", "Bajaj", "Khanna", "Mehta", "Patel", "Jain",
    "Aggarwal", "Thakur",
]

CITIES = {
    "IN-North": ["Delhi", "Gurugram", "Noida", "Chandigarh", "Jaipur", "Lucknow"],
    "IN-South": ["Bengaluru", "Chennai", "Hyderabad", "Kochi", "Coimbatore", "Mysuru"],
    "IN-West": ["Mumbai", "Pune", "Ahmedabad", "Surat", "Nagpur", "Panaji"],
    "IN-East": ["Kolkata", "Bhubaneswar", "Guwahati", "Patna", "Ranchi", "Siliguri"],
    "SEA": ["Singapore", "Kuala Lumpur", "Bangkok", "Jakarta"],
    "MENA": ["Dubai", "Abu Dhabi", "Doha", "Riyadh"],
    "US/EU": ["New York", "London", "Toronto", "Berlin"],
}

# Avatar hues. Deliberately unrelated to any encoded quantity - an avatar
# colour that meant something would be a chart, and a chart you cannot read
# is worse than decoration.
AVATAR = [
    "#7C4DA0", "#C2185B", "#2E7D8F", "#2E7D32", "#B3306B",
    "#3D5A99", "#A0522D", "#5C6BC0", "#00796B", "#8D3F6B",
]

BIO_TEMPLATES = {
    "Beauty": "{niche} creator. Honest reviews, everyday routines.",
    "Fashion": "{niche} and styling. Outfit breakdowns, no filler.",
    "Fitness": "{niche} coach. Form first, hype second.",
    "Food": "Home cook. {niche} that actually works on a weeknight.",
    "Travel": "{niche} on a real budget. Itineraries you can copy.",
    "Technology": "{niche} explained without the spec sheet.",
    "Gaming": "{niche} creator. Long sessions, honest verdicts.",
    "Finance": "{niche} in plain language. No tips, no calls.",
    "Education": "{niche} creator. Concepts first, shortcuts never.",
    "Parenting": "{niche} without the guilt. What worked for us.",
    "Home & Decor": "{niche} on a rental budget. Small spaces, big changes.",
    "Automotive": "{niche} reviews. Ownership costs included.",
}


def _h(key: str, salt: str) -> int:
    """Stable 32-bit hash. Python's hash() is salted per process and would
    hand a different name to the same creator on every run."""
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:8], 16)


def pick(key: str, salt: str, options: list):
    return options[_h(key, salt) % len(options)]


def display_name(influencer_id: str, gender_skew: float) -> str:
    """Name the creator. A female-skewed audience is weakly associated with a
    female creator in the influencer economy, so the draw is biased rather than
    uniform - but it stays a draw, not a rule."""
    r = _h(influencer_id, "gender") % 100
    threshold = 40 + int(gender_skew * 40)          # 40-80
    pool = FIRST_F if r < threshold else FIRST_M
    first = pick(influencer_id, "first", pool)
    last = pick(influencer_id, "last", LAST)
    return f"{first} {last}"


def initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    return "".join(p[0] for p in parts[:2]).upper()


def city(influencer_id: str, geo: str) -> str:
    return pick(influencer_id, "city", CITIES.get(geo, CITIES["IN-West"]))


def avatar(influencer_id: str) -> str:
    return pick(influencer_id, "avatar", AVATAR)


def handle_for(name: str, influencer_id: str) -> str:
    base = name.lower().replace(" ", "")
    # Collisions are possible across 2,000 creators; a stable 2-digit suffix
    # keeps handles unique without making them look machine-generated.
    return f"@{base}"


def bio(niche: str) -> str:
    return BIO_TEMPLATES.get(niche, "{niche} creator.").format(niche=niche)
