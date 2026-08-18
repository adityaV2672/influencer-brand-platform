"""
Central configuration for the Influencer-Brand Collaboration Platform.

Every path, constant and hyper-parameter lives here so that experiments are
reproducible and the dashboard never hard-codes a magic number.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, BENCHMARK_DIR, ARTIFACT_DIR, REPORT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 20260904  # project deadline as the seed, because why not

# --------------------------------------------------------------------------
# Synthetic universe size
# --------------------------------------------------------------------------
N_INFLUENCERS = 2_000
POSTS_PER_INFLUENCER = (12, 40)   # uniform range
N_BRANDS = 120

# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------
NICHES = [
    "Fashion", "Beauty", "Fitness", "Food", "Travel",
    "Technology", "Gaming", "Finance", "Parenting", "Automotive",
    "Home & Decor", "Education",
]

# Follower tiers. Defined in src/data/benchmarks.py so that they match the tier
# boundaries used by the published engagement and pricing sources exactly.
from src.data.benchmarks import TIERS as FOLLOWER_TIERS  # noqa: E402

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ROBERTA_SENTIMENT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_IRONY = "cardiffnlp/twitter-roberta-base-irony"
ROBERTA_EMOTION = "cardiffnlp/twitter-roberta-base-emotion"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct"   # overridden by config if unavailable

# --------------------------------------------------------------------------
# Scoring weights (Phase-1 transparent composite index)
# Weights are documented in the report and validated in benchmark/
# --------------------------------------------------------------------------
PILLAR_WEIGHTS = {
    "reach":      0.20,
    "engagement": 0.35,
    "content":    0.25,
    "network":    0.20,
}

# --------------------------------------------------------------------------
# Freemium gating - single source of truth for the dashboard
# --------------------------------------------------------------------------
FREE_TIER = {
    "max_profile_views":       5,
    "show_numeric_score":      False,
    "show_brand_fit":          False,
    "show_network":            False,
    "show_price_band":         False,
    "advanced_filters":        False,
    "max_shortlists":          1,
}

PAID_TIER = {
    "max_profile_views":       10**9,
    "show_numeric_score":      True,
    "show_brand_fit":          True,
    "show_network":            True,
    "show_price_band":         True,
    "advanced_filters":        True,
    "max_shortlists":          10**9,
}


def tier_of(followers: int) -> str:
    """Map a follower count to its tier label."""
    for lo, hi, label in FOLLOWER_TIERS:
        if lo <= followers < hi:
            return label
    return "Mega"
