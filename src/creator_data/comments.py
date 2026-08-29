"""
The comment corpus - the audience's own voice, as opposed to the creator's.

Why comments are a different signal from captions
--------------------------------------------------
A caption is the creator's marketing copy. It is written to sell, and a
sentiment model run over captions mostly measures how upbeat a creator's
copywriter is. Comments are what the audience wrote back. They carry three
things captions cannot:

  * whether the audience actually responds warmly, or tolerates the creator
  * whether the audience is REAL - bot and engagement-pod comments have a
    recognisable shape, and their share is the best available proxy for
    follower authenticity
  * whether a brand would want its product next to that comment section

Commenter archetypes
--------------------
Each comment is written by one of six archetypes. The mix per creator is
driven by the `authenticity` latent, so a creator with a bought audience gets
a comment section that looks bought. That mix is the ground truth the audience
quality model in audience_quality.py is trained to recover from OBSERVABLE
signals only - it never sees the archetype labels.

SIMULATION NOTE
---------------
The comments are generated. The models applied to them are not: comment_nlp.py
trains its sentiment and offensiveness classifiers on TweetEval, a real corpus
of human-labelled tweets, and applies them here. So the labels on this text are
produced by a model that learned from real human judgements about real short
social text - which is the closest this project can honestly get.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.config import SEED

MAX_COMMENTS_PER_POST = 10

ARCHETYPES = ["genuine_fan", "casual", "engagement_pod", "bot", "critic", "spam"]

# Base mix for a creator with a perfectly authentic audience. The bot, pod and
# spam shares are what `authenticity` moves.
CLEAN_MIX = {"genuine_fan": 0.34, "casual": 0.52, "engagement_pod": 0.04,
             "bot": 0.03, "critic": 0.06, "spam": 0.01}
DIRTY_MIX = {"genuine_fan": 0.10, "casual": 0.24, "engagement_pod": 0.28,
             "bot": 0.28, "critic": 0.04, "spam": 0.06}

TEMPLATES = {
    "genuine_fan": [
        "i tried this after your last video and it actually worked for me",
        "the part about {topic} is what nobody else explains properly",
        "been following since you had 2k, so happy this is working out",
        "please do a full breakdown of {topic}, i keep coming back to this one",
        "genuinely helpful, i sent this to my sister who has the same problem",
        "this answered the exact question i had last week about {topic}",
    ],
    "casual": [
        "love this", "so good", "need this", "obsessed", "this is great",
        "wow", "amazing content", "beautiful", "nice one", "yes exactly",
        "saving this for later", "so pretty", "perfect timing",
    ],
    "engagement_pod": [
        "great post! check out my page too",
        "amazing content as always, support back?",
        "love your feed, following now, follow back",
        "nice! drop a like on my latest",
        "supporting always, do the same",
        "quality content, lets grow together",
    ],
    "bot": [
        "nice", "good", "wow", "super", "first", "cool", "top",
        "follow for follow", "dm for promotion", "check bio for free followers",
        "want more likes? link in bio", "grow your page fast dm now",
    ],
    "critic": [
        "this is just an ad and you did not disclose it properly",
        "did not work for me at all honestly",
        "the price makes this pointless for most people",
        "you said the opposite three months ago",
        "way too sponsored lately, unfollowing",
    ],
    "spam": [
        "earn 5000 daily from home dm me",
        "crypto signals free trial link in bio",
        "buy real followers cheapest rate dm",
        "click my profile for investment tips",
    ],
}

EMOJI = ["🔥", "❤️", "😍", "👏", "💯", "✨", "🙌", "😂", "👌", "💕"]


def _unit(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def archetype_mix(authenticity: float) -> dict:
    """Blend between a clean and a bought-looking comment section."""
    t = float(np.clip((0.95 - authenticity) / 0.65, 0.0, 1.0))
    mix = {k: (1 - t) * CLEAN_MIX[k] + t * DIRTY_MIX[k] for k in ARCHETYPES}
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def generate(posts: pd.DataFrame, latents: pd.DataFrame,
             seed: int = SEED) -> pd.DataFrame:
    """One row per comment.

    The number of comments drawn per post is capped, because the corpus is for
    modelling rather than for reproducing Instagram's exact volume, and the
    creator-level shares - which is what the models actually consume - are
    unbiased by the cap as long as it applies uniformly.
    """
    rng = np.random.default_rng(seed + 7717)
    lat = latents.set_index("influencer_id")["authenticity"].to_dict()

    rows = []
    for post in posts.itertuples():
        iid = str(post.influencer_id)
        auth = float(lat.get(iid, 0.75))
        mix = archetype_mix(auth)
        kinds = list(mix); probs = np.array([mix[k] for k in kinds])

        n_real = int(post.comments or 0)
        n = int(min(n_real, MAX_COMMENTS_PER_POST))
        if n <= 0:
            continue
        drawn = rng.choice(kinds, size=n, p=probs)
        topic = str(getattr(post, "post_niche", "") or "this").lower()

        for j, kind in enumerate(drawn):
            text = str(rng.choice(TEMPLATES[kind])).format(topic=topic)
            # Casual comments carry most of the emoji; bots use them as filler.
            if kind == "casual" and rng.random() < 0.55:
                text = f"{text} {''.join(rng.choice(EMOJI, rng.integers(1, 3)))}"
            elif kind == "bot" and rng.random() < 0.40:
                text = "".join(rng.choice(EMOJI, rng.integers(1, 4)))
            rows.append({
                "comment_id": f"{post.post_id}_C{j:02d}",
                "post_id": post.post_id,
                "influencer_id": iid,
                "text": text,
                "archetype": kind,
                "n_comments_on_post": n_real,
            })

    df = pd.DataFrame(rows)
    # Surface features. These are computed from the text by ordinary code, and
    # they are the ones a real pipeline would also have without any model.
    t = df["text"].astype(str)
    df["n_chars"] = t.str.len()
    df["n_words"] = t.str.split().str.len()
    df["n_emoji"] = t.apply(lambda s: sum(ch in "".join(EMOJI) for ch in s))
    df["is_emoji_only"] = df.n_words.eq(0) | t.str.strip().apply(
        lambda s: len(s) > 0 and all(ch in "".join(EMOJI) + " " for ch in s))
    df["has_link_cue"] = t.str.contains("link in bio|dm|check bio|follow back|"
                                        "follow for follow", case=False, regex=True)
    df["is_generic"] = df.n_words.le(3) & ~df.has_link_cue
    return df


def duplication_rate(comments: pd.DataFrame) -> pd.Series:
    """Share of a creator's comments whose exact text appears more than once.

    A real audience repeats itself a little. A farmed one repeats itself a lot,
    and this needs no model to detect - which is exactly why it belongs in the
    observable feature set rather than in the generator's ground truth.
    """
    g = comments.groupby("influencer_id")["text"]
    return g.apply(lambda s: float(s.duplicated(keep=False).mean()))
