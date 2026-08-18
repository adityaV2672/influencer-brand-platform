"""
Social Network Analysis over the influencer graph.

READ THIS BEFORE QUOTING ANY CENTRALITY NUMBER
----------------------------------------------
The original project design assumed a follower graph. That data does not exist
for third parties: Instagram's Graph API exposes no follower edges, and there is
no legal route to them at project scale. Building the pillar on a follower graph
would have meant either fabricating it silently or dropping the pillar.

Instead the graph is constructed from *observable co-behaviour*:

    co-hashtag edges  - two creators using the same (rare) hashtags
    co-brand edges    - two creators working with or mentioning the same brands

Both are derivable from content alone, which is exactly the data a real platform
would have on day one before any social graph is available.

The consequence, which must be stated wherever these features appear:

    PageRank here measures TOPICAL centrality, not social influence.

A creator with high PageRank sits at the centre of a densely shared thematic
vocabulary. That is a genuinely useful signal for brand matching - it identifies
creators embedded in a category rather than orbiting it - but it is not a claim
about who follows whom, and the dashboard labels it accordingly.

Construction detail: raw co-occurrence counts would make common hashtags
(#reels, #viral) dominate and turn the graph into a hairball. Tags are therefore
TF-IDF weighted so that a shared rare tag counts for much more than a shared
generic one, and the graph is sparsified to a mutual k-nearest-neighbour graph.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import ARTIFACT_DIR, PROCESSED_DIR, SEED

GRAPH_DIR = ARTIFACT_DIR / "network"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# Graph construction
# ==========================================================================


def _tfidf_matrix(
    docs: list[list[str]], min_df: int = 2, max_df_frac: float = 0.35
) -> tuple[sparse.csr_matrix, list[str]]:
    """Build an L2-normalised TF-IDF matrix from token lists."""
    from collections import Counter

    df_counts: Counter = Counter()
    for d in docs:
        df_counts.update(set(d))

    n_docs = len(docs)
    vocab = [
        t for t, c in df_counts.items()
        if c >= min_df and c <= max_df_frac * n_docs
    ]
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    if not vocab:
        return sparse.csr_matrix((n_docs, 0)), []

    idf = np.log((1 + n_docs) / (1 + np.array([df_counts[t] for t in vocab]))) + 1.0

    rows, cols, vals = [], [], []
    for i, d in enumerate(docs):
        tf: Counter = Counter(t for t in d if t in vocab_idx)
        if not tf:
            continue
        total = sum(tf.values())
        for t, c in tf.items():
            j = vocab_idx[t]
            rows.append(i)
            cols.append(j)
            vals.append((c / total) * idf[j])

    m = sparse.csr_matrix((vals, (rows, cols)), shape=(n_docs, len(vocab)))
    norms = np.sqrt(m.multiply(m).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ m, vocab


def _knn_edges(
    sim: np.ndarray, k: int, min_weight: float, mutual: bool = True
) -> list[tuple[int, int, float]]:
    """Sparsify a dense similarity matrix into a k-NN edge list."""
    n = sim.shape[0]
    np.fill_diagonal(sim, -1.0)
    keep = np.zeros((n, n), dtype=bool)
    top = np.argpartition(-sim, kth=min(k, n - 1), axis=1)[:, :k]
    rows = np.repeat(np.arange(n), top.shape[1])
    keep[rows, top.ravel()] = True
    keep &= sim >= min_weight
    adj = (keep & keep.T) if mutual else (keep | keep.T)

    iu = np.triu_indices(n, k=1)
    mask = adj[iu]
    return [
        (int(a), int(b), float(w))
        for a, b, w in zip(iu[0][mask], iu[1][mask], sim[iu][mask])
    ]


def build_graph(
    profiles: pd.DataFrame,
    posts: pd.DataFrame,
    k_hashtag: int = 12,
    k_brand: int = 8,
    min_sim: float = 0.08,
    hashtag_weight: float = 0.65,
    brand_weight: float = 0.35,
) -> nx.Graph:
    """Construct the influencer co-behaviour graph."""
    ids = list(profiles["influencer_id"])
    idx = {v: i for i, v in enumerate(ids)}
    n = len(ids)

    # ---- hashtag documents -------------------------------------------------
    tag_docs: list[list[str]] = [[] for _ in range(n)]
    for inf_id, tags in zip(posts["influencer_id"], posts["hashtags"]):
        i = idx.get(inf_id)
        if i is not None and tags:
            tag_docs[i].extend(str(tags).split("|"))

    # ---- brand documents ---------------------------------------------------
    brand_col = "gen_brand" if "gen_brand" in posts.columns else None
    brand_docs: list[list[str]] = [[] for _ in range(n)]
    if brand_col:
        for inf_id, b in zip(posts["influencer_id"], posts[brand_col]):
            i = idx.get(inf_id)
            if i is not None and isinstance(b, str) and b:
                brand_docs[i].append(b)

    edges: dict[tuple[int, int], float] = {}

    tag_m, tag_vocab = _tfidf_matrix(tag_docs)
    if tag_m.shape[1]:
        sim = (tag_m @ tag_m.T).toarray()
        for a, b, w in _knn_edges(sim, k_hashtag, min_sim):
            edges[(a, b)] = edges.get((a, b), 0.0) + hashtag_weight * w

    brand_m, brand_vocab = _tfidf_matrix(brand_docs, min_df=2, max_df_frac=0.5)
    if brand_m.shape[1]:
        sim = (brand_m @ brand_m.T).toarray()
        for a, b, w in _knn_edges(sim, k_brand, min_sim):
            edges[(a, b)] = edges.get((a, b), 0.0) + brand_weight * w

    G = nx.Graph()
    for i, inf_id in enumerate(ids):
        G.add_node(
            inf_id,
            niche=profiles.iloc[i]["primary_niche"],
            followers=int(profiles.iloc[i]["followers"]),
        )
    for (a, b), w in edges.items():
        G.add_edge(ids[a], ids[b], weight=round(w, 6))

    G.graph.update(
        {
            "construction": "mutual-kNN over TF-IDF co-hashtag and co-brand similarity",
            "k_hashtag": k_hashtag,
            "k_brand": k_brand,
            "min_sim": min_sim,
            "hashtag_vocab": len(tag_vocab),
            "brand_vocab": len(brand_vocab),
            "caveat": "Topical co-behaviour graph, NOT a follower graph. "
                      "Centrality = topical centrality, not social influence.",
        }
    )
    return G


# ==========================================================================
# Centrality
# ==========================================================================


def compute_features(
    G: nx.Graph,
    betweenness_k: int | None = 400,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compute the network features that feed the model and dashboard."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    print(f"  graph: {n:,} nodes, {G.number_of_edges():,} edges, "
          f"density={nx.density(G):.5f}")

    deg = dict(G.degree(weight="weight"))
    deg_unw = dict(G.degree())

    print("  pagerank ...")
    pr = nx.pagerank(G, weight="weight", alpha=0.85, max_iter=200, tol=1e-08)

    print("  eigenvector ...")
    try:
        ev = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:  # noqa: BLE001 - disconnected/degenerate graphs
        ev = {v: float("nan") for v in nodes}

    # Betweenness is O(V*E). Approximate with pivot sampling on large graphs -
    # this is the standard Brandes-Pich estimator and the sample size is
    # reported so the approximation is not hidden.
    if betweenness_k and n > betweenness_k:
        print(f"  betweenness (approx, k={betweenness_k} pivots) ...")
        bt = nx.betweenness_centrality(G, k=betweenness_k, weight=None, seed=seed)
        bt_exact = False
    else:
        print("  betweenness (exact) ...")
        bt = nx.betweenness_centrality(G, weight=None)
        bt_exact = True

    print("  closeness ...")
    cl = nx.closeness_centrality(G)

    print("  louvain communities ...")
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    comm_of = {v: ci for ci, c in enumerate(comms) for v in c}
    comm_size = {ci: len(c) for ci, c in enumerate(comms)}

    print("  clustering ...")
    clus = nx.clustering(G, weight="weight")

    core = nx.core_number(nx.Graph(nx.k_core(G, k=0)))

    df = pd.DataFrame(
        {
            "influencer_id": nodes,
            "degree_centrality": [deg_unw[v] / max(n - 1, 1) for v in nodes],
            "degree_weighted": [deg[v] for v in nodes],
            "pagerank": [pr[v] for v in nodes],
            "eigenvector_centrality": [ev.get(v, np.nan) for v in nodes],
            "betweenness_centrality": [bt[v] for v in nodes],
            "closeness_centrality": [cl[v] for v in nodes],
            "clustering_coefficient": [clus[v] for v in nodes],
            "k_core": [core.get(v, 0) for v in nodes],
            "community": [comm_of.get(v, -1) for v in nodes],
            "community_size": [comm_size.get(comm_of.get(v, -1), 0) for v in nodes],
        }
    )

    # Percentile rank within the whole population - what the dashboard shows,
    # because a raw PageRank of 0.0004 means nothing to a marketing manager.
    for c in ("pagerank", "degree_weighted", "eigenvector_centrality", "betweenness_centrality"):
        df[f"{c}_pct"] = df[c].rank(pct=True)

    df["network_tier"] = pd.cut(
        df["pagerank_pct"],
        bins=[0, 0.50, 0.80, 0.95, 1.01],
        labels=["Peripheral", "Connected", "Influential", "Hub"],
        right=False,
    ).astype(str)

    df.attrs["betweenness_exact"] = bt_exact
    df.attrs["betweenness_k"] = betweenness_k
    return df


# ==========================================================================


def run(save: bool = True) -> pd.DataFrame:
    profiles = pd.read_parquet(PROCESSED_DIR / "profiles.parquet")
    posts = pd.read_parquet(PROCESSED_DIR / "posts.parquet")

    print("Building influencer co-behaviour graph")
    G = build_graph(profiles, posts)
    feats = compute_features(G)

    if save:
        nx.write_gexf(G, GRAPH_DIR / "influencer_graph.gexf")
        feats.to_parquet(GRAPH_DIR / "network_features.parquet", index=False)

        # Trimmed edge list for the dashboard's network map - the full graph is
        # too heavy to render client-side.
        edges = nx.to_pandas_edgelist(G).sort_values("weight", ascending=False)
        edges.head(6000).to_parquet(GRAPH_DIR / "edges_top.parquet", index=False)

        (GRAPH_DIR / "graph_meta.json").write_text(
            json.dumps(
                {
                    **{k: (v if isinstance(v, (int, float, str)) else str(v))
                       for k, v in G.graph.items()},
                    "n_nodes": G.number_of_nodes(),
                    "n_edges": G.number_of_edges(),
                    "density": nx.density(G),
                    "n_communities": int(feats["community"].nunique()),
                    "largest_community": int(feats["community_size"].max()),
                    "betweenness_exact": bool(feats.attrs.get("betweenness_exact")),
                    "betweenness_pivots": feats.attrs.get("betweenness_k"),
                },
                indent=2,
            )
        )
        print(f"  saved to {GRAPH_DIR}")

    return feats


if __name__ == "__main__":
    f = run()
    print("\nnetwork tier distribution:")
    print(f["network_tier"].value_counts().to_string())
    print(f"\ncommunities: {f['community'].nunique()}  "
          f"(largest {f['community_size'].max()} members)")
    print("\ntop 10 by PageRank:")
    print(f.nlargest(10, "pagerank")[
        ["influencer_id", "pagerank", "degree_centrality", "community", "network_tier"]
    ].to_string(index=False))
