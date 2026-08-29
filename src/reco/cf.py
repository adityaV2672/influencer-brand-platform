"""
Collaborative filtering over the brand-creator interaction matrix.

The gap this is here to fill
----------------------------
ranker.py measured it: on a brand the model has never seen, learning to rank
from creator FEATURES does not beat hand-set weights. That is not a failure of
the ranker. Per-brand taste is not a function of the creator's attributes, so
no model that sees only attributes can recover it - the information is in who
the brand has previously chosen, and nowhere else.

Collaborative filtering uses exactly that. Brands that shortlisted similar
creators get similar factors, and a creator one of them signed becomes a
candidate for the others.

Honest about the cold start
---------------------------
A brand with no history has no row in the matrix and CF can say nothing about
it. That is not a bug to engineer around; it is why the platform must lead with
content-based scoring and blend CF in as history accumulates. `hybrid()` does
exactly that and the blend weight is reported.

Evaluation is leave-one-out per brand: hold out one creator the brand actually
engaged, rank the rest of the catalogue, and ask where the held-out creator
lands. Compared against popularity (recommend whoever is most shortlisted
overall), which is the baseline any recommender has to beat before it is worth
deploying.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.config import ARTIFACT_DIR, SEED

OUT = ARTIFACT_DIR / "reco"
RESULTS = OUT / "cf_results.json"

N_FACTORS = 24
# Implicit feedback is not binary: contacting a creator says more than viewing
# one, and completing a deal says most.
STAGE_WEIGHT = {"viewed": 0.0, "shortlisted": 1.0, "contacted": 2.0,
                "accepted": 3.0, "completed": 4.0}
MIN_EVENTS = 3


def matrix(log: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    d = log.copy()
    d["w"] = d.stage.map(STAGE_WEIGHT).fillna(0.0)
    d = d[d.w > 0]
    brands = sorted(d.brand_id.unique())
    creators = sorted(d.influencer_id.unique())
    bi = {b: i for i, b in enumerate(brands)}
    ci = {c: i for i, c in enumerate(creators)}
    M = np.zeros((len(brands), len(creators)), dtype="float32")
    for r in d.itertuples():
        M[bi[r.brand_id], ci[r.influencer_id]] = max(M[bi[r.brand_id], ci[r.influencer_id]], r.w)
    return M, brands, creators


def _rank_of(scores: np.ndarray, target: int, mask: np.ndarray) -> int:
    s = scores.copy()
    s[mask] = -np.inf          # exclude the brand's other known positives
    s[target] = scores[target]
    return int((s > scores[target]).sum()) + 1


def evaluate(M: np.ndarray, k_list=(10, 20, 50), n_factors: int = N_FACTORS) -> dict:
    """Leave-one-out per brand, against a popularity baseline."""
    rng = np.random.default_rng(SEED)
    popularity = (M > 0).sum(axis=0).astype(float)

    hits = {f"cf_hit@{k}": 0 for k in k_list}
    hits.update({f"pop_hit@{k}": 0 for k in k_list})
    ranks_cf, ranks_pop, n = [], [], 0

    for b in range(M.shape[0]):
        pos = np.flatnonzero(M[b] > 0)
        if len(pos) < MIN_EVENTS:
            continue
        held = int(rng.choice(pos))
        train = M.copy()
        train[b, held] = 0.0

        svd = TruncatedSVD(n_components=min(n_factors, min(train.shape) - 1),
                           random_state=SEED)
        U = svd.fit_transform(train)
        recon = U @ svd.components_

        others = np.array([p for p in pos if p != held])
        mask = np.zeros(M.shape[1], dtype=bool)
        mask[others] = True

        r_cf = _rank_of(recon[b], held, mask)
        r_pop = _rank_of(popularity, held, mask)
        ranks_cf.append(r_cf); ranks_pop.append(r_pop); n += 1
        for k in k_list:
            hits[f"cf_hit@{k}"] += int(r_cf <= k)
            hits[f"pop_hit@{k}"] += int(r_pop <= k)

    res = {
        "n_brands_evaluated": n,
        "n_creators": int(M.shape[1]),
        "n_factors": int(n_factors),
        "sparsity": round(float((M > 0).mean()), 5),
        "median_rank_cf": int(np.median(ranks_cf)) if ranks_cf else None,
        "median_rank_popularity": int(np.median(ranks_pop)) if ranks_pop else None,
        **{k: round(v / max(n, 1), 4) for k, v in hits.items()},
        "caveats": {
            "cold_start": "A brand with no history has no row here and CF returns "
                          "nothing for it. Content-based scoring carries the "
                          "product until history exists; hybrid() is the blend.",
            "data": "SIMULATED interaction log (src/reco/interactions.py).",
        },
    }
    RESULTS.write_text(json.dumps(res, indent=2))
    return res


def factors(M: np.ndarray, n_factors: int = N_FACTORS) -> np.ndarray:
    svd = TruncatedSVD(n_components=min(n_factors, min(M.shape) - 1),
                       random_state=SEED)
    return svd.fit_transform(M) @ svd.components_


def hybrid(cf_scores: np.ndarray, content_scores: np.ndarray,
           n_events: int, full_weight_at: int = 40) -> tuple[np.ndarray, float]:
    """Blend CF into content scoring in proportion to how much history exists.

    A brand with three events should barely feel the collaborative signal; one
    with forty should feel it fully. Ramping on the brand's own event count is
    what stops a two-interaction brand being recommended somebody else's taste.
    """
    w = float(np.clip(n_events / full_weight_at, 0.0, 1.0)) * 0.5

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    return (1 - w) * z(content_scores) + w * z(cf_scores), round(w, 3)
