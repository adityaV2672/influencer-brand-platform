"""
Publication figures for the written report and the slide deck.

Every figure is generated from the artifacts on disk. Nothing is hand-drawn and
no number is typed in by hand, so re-running the pipeline regenerates a report
whose figures and text agree with each other by construction.

Colour discipline matches the dashboard: fixed categorical slot order, one-hue
sequential ramps, reserved status colours, recessive grid, no dual axes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from src.config import ARTIFACT_DIR, FIGURE_DIR, PROCESSED_DIR

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e6e5e1"
FAMILY_COLOR = {
    "baseline": MUTED, "lexicon": SERIES[1], "classical-ml": SERIES[3],
    "transformer": SERIES[0], "llm": SERIES[6],
}

plt.rcParams.update({
    "figure.dpi": 190,
    "savefig.dpi": 190,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})


def _style(ax, ytitle="", xtitle="", title="", grid_axis="y"):
    ax.set_title(title, loc="left", pad=9)
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, linewidth=0.7, color=GRID)
    ax.set_axisbelow(True)
    return ax


def _save(fig, name: str) -> Path:
    p = FIGURE_DIR / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return json.loads(path.read_text())


# ==========================================================================
# 1. Benchmark calibration
# ==========================================================================


def fig_calibration() -> Path | None:
    from src.data import benchmarks as bm

    prof = _load(PROCESSED_DIR / "profiles.parquet")
    if prof is None:
        return None
    order = ["Nano", "Micro", "Mid", "Macro", "Mega"]
    present = [t for t in order if (prof["follower_tier"] == t).any()]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    x = np.arange(len(present))
    lo = [bm.ENGAGEMENT_BANDS[t][0] * 100 for t in present]
    hi = [bm.ENGAGEMENT_BANDS[t][1] * 100 for t in present]
    med = [prof[prof["follower_tier"] == t]["engagement_rate"].median() * 100 for t in present]
    ax.bar(x, np.array(hi) - np.array(lo), bottom=lo, width=0.62,
           color=SEQ[1], edgecolor="white", linewidth=1.2, label="Published band")
    ax.plot(x, med, "o", color=SERIES[1], markersize=8, markeredgecolor="white",
            markeredgewidth=1.4, label="Synthetic median", zorder=5)
    ax.set_xticks(x, present)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _style(ax, ytitle="Engagement rate", title="Engagement calibration by tier")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    lo = [bm.PRICE_BANDS_INR[t][0] for t in present]
    hi = [bm.PRICE_BANDS_INR[t][1] for t in present]
    camp = _load(PROCESSED_DIR / "campaigns.parquet")
    med = []
    if camp is not None:
        m = camp.merge(prof[["influencer_id", "follower_tier"]], on="influencer_id")
        med = [m[m["follower_tier"] == t]["fee_inr"].median() for t in present]
    ax.bar(x, np.array(hi) - np.array(lo), bottom=lo, width=0.62,
           color=SEQ[1], edgecolor="white", linewidth=1.2, label="Published band")
    if len(med):
        ax.plot(x, med, "o", color=SERIES[1], markersize=8, markeredgecolor="white",
                markeredgewidth=1.4, label="Synthetic median", zorder=5)
    ax.set_yscale("log")
    ax.set_xticks(x, present)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"₹{v/1e7:.0f}Cr" if v >= 1e7 else f"₹{v/1e5:.0f}L" if v >= 1e5 else f"₹{v/1e3:.0f}K"))
    _style(ax, ytitle="Fee per deliverable (log)", title="Fee calibration by tier")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    return _save(fig, "fig01_calibration")


# ==========================================================================
# 2. NLP method comparison
# ==========================================================================


def fig_nlp_comparison(
    task: str, corpus: str | None = None, name: str = "", filename: str | None = None
) -> Path | None:
    res = _load(ARTIFACT_DIR / "benchmarks" / "results.parquet")
    if res is None:
        return None
    t = res[(res["status"] == "ok") & (res["task"] == task)]
    if corpus:
        t = t[t["corpus"] == corpus]
    if not len(t):
        return None
    t = t.sort_values("macro_f1")

    fig, ax = plt.subplots(figsize=(8.2, max(2.6, 0.44 * len(t) + 1.0)))
    colors = [FAMILY_COLOR.get(f, MUTED) for f in t["family"]]
    bars = ax.barh(t["method_name"], t["macro_f1"], color=colors, height=0.66)
    for b, v in zip(bars, t["macro_f1"]):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8.5, color=INK2)

    base = t[t["method_key"] == "majority_baseline"]
    if len(base):
        ax.axvline(float(base["macro_f1"].iloc[0]), color=INK, linewidth=1,
                   linestyle=":", zorder=1)
        ax.text(float(base["macro_f1"].iloc[0]), len(t) - 0.35, " majority baseline",
                fontsize=7.6, color=INK2, va="top")

    ax.set_xlim(0, min(1.06, float(t["macro_f1"].max()) * 1.24))
    _style(ax, xtitle="Macro-F1", title=name or f"{task.title()} — method comparison", grid_axis="x")

    seen, handles = [], []
    for f in ["baseline", "lexicon", "classical-ml", "transformer", "llm"]:
        if f in set(t["family"]) and f not in seen:
            seen.append(f)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]))
    ax.legend(handles, seen, frameon=False, fontsize=8, ncol=len(seen),
              loc="lower right", bbox_to_anchor=(1.0, -0.02))
    fig.tight_layout()
    # Stable filenames: the report and the deck reference these by name, so they
    # must not vary with which corpus was passed.
    return _save(fig, filename or f"fig_nlp_{task}")


def fig_irony_crossdomain() -> Path | None:
    """The cross-domain test: does a method trained on tweets survive on headlines?"""
    res = _load(ARTIFACT_DIR / "benchmarks" / "results.parquet")
    if res is None:
        return None
    t = res[(res["status"] == "ok") & (res["task"] == "irony")]
    corpora = sorted(t["corpus"].unique())
    if len(corpora) < 2:
        return None

    piv = t.pivot_table(index="method_name", columns="corpus", values="macro_f1")
    piv = piv.dropna().sort_values(corpora[0])
    if not len(piv):
        return None

    fam = t.drop_duplicates("method_name").set_index("method_name")["family"]
    y = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(8.4, max(2.8, 0.5 * len(piv) + 1.0)))
    ax.barh(y - 0.19, piv[corpora[0]], height=0.36, color=SERIES[0], label=corpora[0])
    ax.barh(y + 0.19, piv[corpora[1]], height=0.36, color=SERIES[1], label=corpora[1])
    ax.set_yticks(y, piv.index)
    _style(ax, xtitle="Macro-F1", title="Irony detection — in-domain vs cross-domain", grid_axis="x")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, "fig_irony_crossdomain")


# ==========================================================================
# 3. Model results
# ==========================================================================


def fig_model_vs_baselines() -> Path | None:
    r = _load(ARTIFACT_DIR / "models" / "model_results.json")
    if r is None:
        return None
    p = r["performance"]
    rows = [
        ("Published benchmark curve\n(no learning)", p["baseline_benchmark_curve"]["r2_log"], MUTED),
        ("Phase-1 weighted index\n(proposal design)", p["baseline_composite_index"]["r2_log"], MUTED),
        ("LightGBM\n(this system)", p["r2_log"], SERIES[0]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ax.bar(names, vals, color=[r[2] for r in rows], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:.3f}",
                ha="center", fontsize=9.5, color=INK, fontweight="semibold")
    ceiling = p.get("theoretical_r2_log_ceiling")
    if ceiling:
        ax.axhline(ceiling, color=INK, linestyle=":", linewidth=1)
        ax.text(2.42, ceiling, f" ceiling {ceiling:.2f}", fontsize=7.8, color=INK2, va="bottom", ha="right")
    ax.axhline(0, color=INK, linewidth=0.9)
    _style(ax, ytitle="R² (log), out-of-fold", title="Does the learned model earn its complexity?")
    fig.tight_layout()
    return _save(fig, "fig_model_baselines")


def fig_ablation() -> Path | None:
    r = _load(ARTIFACT_DIR / "models" / "model_results.json")
    if r is None:
        return None
    drops = r["performance"].get("ablation", {}).get("drops", {})
    if not drops:
        return None
    names = list(drops)
    deltas = [drops[k]["delta"] for k in names]
    order = np.argsort(deltas)
    names = [names[i].title() for i in order]
    deltas = [deltas[i] for i in order]
    colors = [SERIES[0] if d > 0 else SERIES[7] for d in deltas]

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.barh(names, deltas, color=colors, height=0.6)
    for i, d in enumerate(deltas):
        ax.text(d + (0.003 if d >= 0 else -0.003), i, f"{d:+.3f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK, linewidth=0.9)
    _style(ax, xtitle="R² lost when the pillar is removed",
           title="Which feature pillar actually contributes?", grid_axis="x")
    fig.tight_layout()
    return _save(fig, "fig_ablation")


def fig_importance(top_n: int = 15) -> Path | None:
    imp = _load(ARTIFACT_DIR / "models" / "performance_importance.parquet")
    if imp is None:
        return None
    d = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(d) + 1.1))
    ax.barh(d["feature"], d["gain_pct"], color=SERIES[0], height=0.66)
    for i, v in enumerate(d["gain_pct"]):
        ax.text(v + 0.35, i, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    _style(ax, xtitle="Share of total split gain (%)",
           title="What the model relies on", grid_axis="x")
    fig.tight_layout()
    return _save(fig, "fig_importance")


def fig_pred_vs_actual() -> Path | None:
    oof = ARTIFACT_DIR / "models" / "performance_oof.npy"
    tbl = ARTIFACT_DIR / "features" / "modelling_table.parquet"
    if not oof.exists() or not tbl.exists():
        return None
    pred = np.load(oof)
    y = pd.read_parquet(tbl)["campaign_engagement_rate"].to_numpy()
    n = min(len(pred), len(y))
    pred, y = pred[:n], y[:n]

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(y * 100, pred * 100, s=11, alpha=0.4, color=SERIES[0],
               edgecolor="white", linewidth=0.35)
    lim = [min(y.min(), pred.min()) * 100 * 0.85, max(y.max(), pred.max()) * 100 * 1.1]
    ax.plot(lim, lim, color=INK, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    _style(ax, xtitle="Actual campaign engagement (%)", ytitle="Predicted (%)",
           title="Out-of-fold predictions", grid_axis="both")
    fig.tight_layout()
    return _save(fig, "fig_pred_vs_actual")


# ==========================================================================
# 4. Topics and network
# ==========================================================================


def fig_topic_coherence() -> Path | None:
    coh = _load(ARTIFACT_DIR / "topics" / "coherence.json")
    if coh is None:
        return None
    met = ["npmi", "c_v", "diversity"]
    x = np.arange(len(met))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, (label, key) in enumerate([("BERTopic", "bertopic"), ("LDA", "lda")]):
        vals = [coh[key][m] for m in met]
        ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=SERIES[i], label=label,
               edgecolor="white", linewidth=1.2)
        for xi, v in zip(x + (i - 0.5) * 0.36, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [m.upper() for m in met])
    ax.axhline(0, color=INK, linewidth=0.9)
    _style(ax, ytitle="Score", title="Topic quality — BERTopic vs LDA")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "fig_topic_coherence")


def fig_network_degree() -> Path | None:
    net = _load(ARTIFACT_DIR / "network" / "network_features.parquet")
    if net is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    ax = axes[0]
    deg = (net["degree_centrality"] * (len(net) - 1)).round().astype(int)
    vc = deg.value_counts().sort_index()
    ax.scatter(vc.index, vc.values, s=16, color=SERIES[0], alpha=0.75,
               edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, xtitle="Degree (log)", ytitle="Creators (log)",
           title="Degree distribution", grid_axis="both")

    ax = axes[1]
    order = ["Peripheral", "Connected", "Influential", "Hub"]
    counts = net["network_tier"].value_counts().reindex(order).fillna(0)
    ax.bar(order, counts.values, color=[MUTED, SEQ[2], SERIES[0], SERIES[6]], width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.02, f"{int(v):,}", ha="center", fontsize=8.5, color=INK2)
    _style(ax, ytitle="Creators", title="Network position tiers")

    fig.tight_layout()
    return _save(fig, "fig_network")


def fig_engagement_decay() -> Path | None:
    prof = _load(PROCESSED_DIR / "profiles.parquet")
    if prof is None:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    s = prof.sample(n=min(1400, len(prof)), random_state=3)
    ax.scatter(s["followers"], s["engagement_rate"] * 100, s=9, alpha=0.32,
               color=SERIES[0], edgecolor="none")
    from src.data import benchmarks as bm

    xs = np.logspace(3, 7, 120)
    ax.plot(xs, [bm.expected_er(x) * 100 for x in xs], color=SERIES[1], linewidth=2,
            label="Published benchmark curve")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    _style(ax, xtitle="Followers (log)", ytitle="Engagement rate (log)",
           title="Why follower count is a poor ranking signal", grid_axis="both")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "fig_engagement_decay")


# ==========================================================================


def build_all() -> dict[str, str]:
    out: dict[str, str] = {}
    jobs = [
        ("calibration", fig_calibration),
        ("engagement_decay", fig_engagement_decay),
        ("nlp_sentiment", lambda: fig_nlp_comparison(
            "sentiment", "tweeteval_sentiment",
            "Sentiment — method comparison (TweetEval)", "fig_nlp_sentiment")),
        ("nlp_irony", lambda: fig_nlp_comparison(
            "irony", "tweeteval_irony",
            "Irony — method comparison (TweetEval)", "fig_nlp_irony")),
        ("nlp_sarcasm_headlines", lambda: fig_nlp_comparison(
            "irony", "sarcasm_headlines",
            "Sarcasm — method comparison (news headlines)", "fig_nlp_sarcasm_headlines")),
        ("nlp_emotion", lambda: fig_nlp_comparison(
            "emotion", "tweeteval_emotion",
            "Emotion — method comparison (TweetEval)", "fig_nlp_emotion")),
        ("irony_crossdomain", fig_irony_crossdomain),
        ("model_baselines", fig_model_vs_baselines),
        ("ablation", fig_ablation),
        ("importance", fig_importance),
        ("pred_vs_actual", fig_pred_vs_actual),
        ("topic_coherence", fig_topic_coherence),
        ("network", fig_network_degree),
    ]
    for name, fn in jobs:
        try:
            p = fn()
            if p:
                out[name] = str(p)
                print(f"    {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"    {name} FAILED: {type(exc).__name__}: {exc}")
    return out


if __name__ == "__main__":
    print("building figures ...")
    figs = build_all()
    print(f"{len(figs)} figures in {FIGURE_DIR}")
