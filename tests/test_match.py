"""
The typed-brief matcher.

The intake page is the one surface where a brand's own words drive the result,
so the things worth pinning down are: does it use the same composite as the
batch engine, does the competitor veto actually veto, does the budget gate
actually gate, and does it fail honestly when the words land on nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from nectar import match  # noqa: E402


def _brief(**kw) -> "match.Brief":
    base = dict(
        category="Beauty",
        brand_text="Affordable everyday skincare and makeup, simple routines.",
        campaign_text="A new moisturiser shown in a real morning routine.",
        competitors=[], geos=["IN-West"], ages=["18-24"],
        min_followers=0, budget=750_000, cap=10 ** 9,
        n_reel=2, n_story=3, n_carousel=0, objective="Consideration",
    )
    base.update(kw)
    return match.Brief(**base)


# --------------------------------------------------------------------------
def test_weights_match_the_batch_engine():
    """The intake page must not quietly become a second scoring system."""
    from src.models.brandfit import COMPONENT_WEIGHTS as BATCH
    assert match.COMPONENT_WEIGHTS == BATCH


def test_weights_sum_to_one():
    assert sum(match.COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)


def test_scores_every_creator():
    from nectar import data
    ranked, info = match.score(_brief())
    assert len(ranked) == len(data.creators())
    assert info["pool"] == len(ranked)
    assert ranked["rank"].is_monotonic_increasing


def test_components_are_bounded():
    ranked, _ = match.score(_brief())
    for col in ("fit_semantic_similarity", "fit_category_match",
                "fit_audience_match", "fit_content_safety", "fit_consistency"):
        assert ranked[col].between(0.0, 1.0).all(), col
    assert ranked["fit"].between(0.0, 1.0).all()


def test_matched_and_ignored_words_are_reported():
    _, info = match.score(_brief())
    assert "skincare" in info["matched"]
    # A word that cannot be in a creator caption must be reported, not dropped.
    _, info2 = match.score(_brief(brand_text="quixotic perambulation skincare"))
    assert "quixotic" in info2["ignored"]


def test_unmatched_brief_falls_back_instead_of_ranking_on_noise():
    """A brief made of words no creator uses must not silently reorder people."""
    b = _brief(brand_text="zzqq wubbleflux", campaign_text="grondlewick")
    ranked, info = match.score(b)
    assert info["fallback"] is True
    assert (ranked["fit_semantic_similarity"] == 0.5).all()


def test_competitor_conflict_blocks_rather_than_deducts():
    mentions = match.brand_mentions()
    if mentions.empty:
        pytest.skip("no brand-mention evidence exported")
    top = mentions.loc[mentions.n_paid.idxmax(), "brand"]
    clean, _ = match.score(_brief(competitors=[]))
    gated, info = match.score(_brief(competitors=[top]))
    assert info["blocked"] > 0
    assert (gated["gate"] == 0).sum() > (clean["gate"] == 0).sum()
    blocked = gated[gated.gate == 0]
    assert (blocked["fit"] == 0).all()
    assert not blocked["eligible"].any()


def test_budget_cap_and_audience_floor_are_hard_gates():
    ranked, _ = match.score(_brief(cap=40_000, min_followers=50_000))
    ok = ranked[ranked.eligible]
    assert (ok.fee <= 40_000).all()
    assert (ok.followers >= 50_000).all()


def test_goal_changes_the_ranking():
    reach, _ = match.score(_brief(objective="Awareness"))
    rate, _ = match.score(_brief(objective="Conversion"))
    top_reach = list(reach.head(20).influencer_id)
    top_rate = list(rate.head(20).influencer_id)
    assert top_reach != top_rate, "the campaign goal is not affecting the order"


def test_fit_is_the_dominant_term():
    """The goal may reorder within a fit band; it must not override fit."""
    assert match.GOAL_WEIGHT < 0.5


def test_reasons_are_plain_and_non_empty_for_the_top_row():
    b = _brief()
    ranked, info = match.score(b)
    top = ranked.iloc[0]
    text = match.reasons(top, b, info)
    assert text and all(isinstance(t, str) and t for t in text)


def test_lexical_tables_are_small_enough_to_ship():
    """The hosted app is a parquet reader; these must not become a payload."""
    app_data = ROOT / "app_data"
    for name, limit_kb in [("nectar_vocab.parquet", 200),
                           ("nectar_creator_terms.parquet", 3000),
                           ("nectar_brand_mentions.parquet", 500)]:
        p = app_data / name
        assert p.exists(), f"{name} missing - run export_nectar"
        assert p.stat().st_size / 1024 < limit_kb, name


def test_vocabulary_is_consistent_with_the_creator_terms():
    vocab = set(match.vocab()["term"])
    terms = match.creator_terms()
    assert set(terms["term"]) <= vocab
    # Profiles are L2-normalised after truncation, so no cosine can exceed 1.
    norms = (terms.assign(sq=terms.weight ** 2)
             .groupby("influencer_id")["sq"].sum() ** 0.5)
    assert norms.between(0.99, 1.01).all()
