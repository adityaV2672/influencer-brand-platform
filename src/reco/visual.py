"""
Visual embeddings for creator feeds.

SIMULATED ENCODER, REAL HEAD - the same split as the audio work
----------------------------------------------------------------
There are no images in this project. A CLIP or DINOv2 encoder was not run.
What is generated is an embedding SPACE with the property a downstream model
depends on: a creator's visual identity is a stable point, category signature
is linearly decodable, and per-post variation around that point is larger for
creators who post inconsistently.

The head on top is real. `evaluate()` trains a classifier to recover a
creator's niche from their visual centroid alone, under cross-validation, and
reports what it gets. If the embedding space did not encode category, that
number would collapse - so it is a check on the simulation as much as a
result.

What this buys the product: brands describe a look ("clean, minimal, daylight")
that no caption contains. Visual similarity is the only component of brand fit
that can see it.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

from src.config import ARTIFACT_DIR, SEED

DIM = 64
OUT = ARTIFACT_DIR / "visual"
OUT.mkdir(parents=True, exist_ok=True)

# Named visual attributes a brand can actually ask for. Each is a direction in
# the space, so "show me bright, minimal feeds" is a projection rather than a
# keyword search.
ATTRIBUTES = ["brightness", "saturation", "minimalism", "people_present",
              "product_focus", "outdoor"]


def _unit(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def _basis(seed: int = SEED) -> np.ndarray:
    """Fixed orthonormal directions: category, attributes, then nuisance."""
    rng = np.random.default_rng(seed + 606)
    m = rng.normal(0, 1, (DIM, DIM))
    q, _ = np.linalg.qr(m)
    return q


def creator_embeddings(creators: pd.DataFrame, latents: pd.DataFrame,
                       seed: int = SEED) -> tuple[np.ndarray, pd.DataFrame]:
    """One visual centroid per creator, plus the readable attribute scores."""
    rng = np.random.default_rng(seed + 1414)
    d = creators.copy()
    d["influencer_id"] = d.influencer_id.astype(str)
    lat = latents.copy(); lat["influencer_id"] = lat.influencer_id.astype(str)
    d = d.merge(lat[["influencer_id", "content_quality", "consistency", "niche_focus"]],
                on="influencer_id", how="left")

    niches = sorted(d.primary_niche.dropna().unique())
    basis = _basis(seed)
    n = len(d)

    # Category signature: every creator in a niche shares a direction, scaled
    # by how tightly they stick to that niche.
    cat_idx = d.primary_niche.map({k: i for i, k in enumerate(niches)}).fillna(0).astype(int)
    cat_vec = basis[cat_idx.to_numpy()] * (0.6 + 0.9 * d.niche_focus.fillna(0.6).to_numpy())[:, None]

    # Readable attributes, each on its own direction.
    attrs = {}
    attr_component = np.zeros((n, DIM))
    for k, name in enumerate(ATTRIBUTES):
        v = np.array([_unit(str(i), f"vis_{name}") for i in d.influencer_id])
        # Aesthetic quality lifts brightness and minimalism a little; the rest
        # are independent style choices.
        if name in ("brightness", "minimalism"):
            v = np.clip(0.65 * v + 0.35 * d.content_quality.fillna(0.5).to_numpy(), 0, 1)
        attrs[f"visual_{name}"] = np.round(v, 4)
        attr_component += np.outer(v - 0.5, basis[len(niches) + k]) * 1.1

    nuisance = rng.normal(0, 0.55, (n, DIM))
    emb = cat_vec + attr_component + nuisance
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    # Feed coherence: how tightly a creator's posts cluster around their own
    # centroid. Low coherence is a real brand risk - the brand approves a look
    # and gets something else.
    coherence = np.clip(0.35 + 0.55 * d.consistency.fillna(0.6).to_numpy()
                        + rng.normal(0, 0.06, n), 0.05, 0.99)

    meta = pd.DataFrame({"influencer_id": d.influencer_id.to_numpy(),
                         "visual_coherence": np.round(coherence, 4), **attrs})
    return emb.astype("float32"), meta


def brand_visual_profile(brands: pd.DataFrame, creators: pd.DataFrame,
                         emb: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """A brand's look, as the average of the feeds in its own category.

    Deliberately not a free-floating vector: a brand's visual target has to be
    expressible in the same space as the creators, or the cosine means nothing.
    """
    ids = list(meta.influencer_id)
    index = {k: i for i, k in enumerate(ids)}
    c = creators.copy(); c["influencer_id"] = c.influencer_id.astype(str)
    out = np.zeros((len(brands), emb.shape[1]), dtype="float32")
    for bi, brand in enumerate(brands.itertuples()):
        peers = c[c.primary_niche == getattr(brand, "category", None)]
        rows = [index[i] for i in peers.influencer_id if i in index]
        v = emb[rows].mean(axis=0) if rows else emb.mean(axis=0)
        out[bi] = v / (np.linalg.norm(v) + 1e-9)
    return out


def evaluate(emb: np.ndarray, creators: pd.DataFrame) -> dict:
    """Can a real model read category off the simulated embeddings?"""
    y = creators.primary_niche.astype(str).to_numpy()
    clf = LogisticRegression(max_iter=2000, C=2.0, random_state=SEED)
    pred = cross_val_predict(clf, emb, y, cv=5)
    majority = np.full(len(y), pd.Series(y).value_counts().idxmax())
    res = {
        "task": "recover creator niche from the visual centroid alone",
        "n_creators": int(len(y)), "n_classes": int(len(set(y))), "dim": int(emb.shape[1]),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "majority_accuracy": round(float((majority == y).mean()), 4),
        "caveats": {
            "encoder": "SIMULATED. No image was loaded; CLIP and DINOv2 were not "
                       "run. The space is constructed so category is linearly "
                       "decodable, so this score confirms the construction "
                       "worked - it is NOT evidence that visual models can "
                       "identify creator niche.",
            "head": "REAL. Logistic regression, 5-fold cross-validated.",
        },
    }
    (OUT / "visual_results.json").write_text(json.dumps(res, indent=2))
    return res
