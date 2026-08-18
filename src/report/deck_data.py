"""
Extract the numbers the slide deck needs into one JSON file.

The deck generator (build_deck.js) reads only this file and the figure PNGs.
Nothing on a slide is typed by hand, so the deck cannot drift out of agreement
with the report or with the artifacts.

If a stage did not run, the corresponding value becomes an explicit
"not available in this build" string rather than a plausible-looking number.
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, FIGURE_DIR, PROCESSED_DIR, REPORT_DIR

DECK_DIR = REPORT_DIR / "deck"
NA = "n/a"


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _p(p: Path):
    return pd.read_parquet(p) if p.exists() else None


def _pct(x, dp: int = 2) -> str:
    return NA if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x) * 100:.{dp}f}%"


def build() -> Path:
    DECK_DIR.mkdir(parents=True, exist_ok=True)
    figdir = DECK_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for png in FIGURE_DIR.glob("*.png"):
        shutil.copy(png, figdir / png.name)

    prof = _p(PROCESSED_DIR / "profiles.parquet")
    posts = _p(PROCESSED_DIR / "posts.parquet")
    camps = _p(PROCESSED_DIR / "campaigns.parquet")
    bench = _p(ARTIFACT_DIR / "benchmarks" / "results.parquet")
    models = _j(ARTIFACT_DIR / "models" / "model_results.json")
    coh = _j(ARTIFACT_DIR / "topics" / "coherence.json")

    d: dict = {
        "author": "Aditya Verma",
        "date": date.today().strftime("%d %B %Y"),
    }

    # ---- problem slide ----------------------------------------------------
    if prof is not None:
        nano = prof[prof["follower_tier"] == "Nano"]["engagement_rate"].median()
        macro_pool = prof[prof["follower_tier"].isin(["Macro", "Mega"])]["engagement_rate"]
        macro = macro_pool.median() if len(macro_pool) >= 5 else \
            prof[prof["follower_tier"] == "Mid"]["engagement_rate"].median()
        d["problem"] = {
            "nano_er": _pct(nano, 1),
            "macro_er": _pct(macro, 1),
            "ratio": f"{nano / macro:.1f}×" if macro and macro > 0 else NA,
        }
    else:
        d["problem"] = {"nano_er": NA, "macro_er": NA, "ratio": NA}

    d["data"] = {
        "n_influencers": f"{len(prof):,}" if prof is not None else NA,
        "n_posts": f"{len(posts):,}" if posts is not None else NA,
        "n_campaigns": f"{len(camps):,}" if camps is not None else NA,
    }

    # ---- sentiment --------------------------------------------------------
    d["sentiment"] = {"finding": "Sentiment benchmark did not run in this build.", "n_eval": NA}
    if bench is not None:
        s = bench[(bench["status"] == "ok") & (bench["task"] == "sentiment")]
        if len(s):
            best = s.sort_values("macro_f1", ascending=False).iloc[0]
            nrc = s[s["method_name"].str.contains("NRC", case=False)]
            vad = s[s["method_name"].str.contains("VADER", case=False)]
            parts = [f"Best method: {best['method_name']} at macro-F1 {best['macro_f1']:.3f}."]
            if len(nrc) and len(vad):
                nf, vf = float(nrc["macro_f1"].iloc[0]), float(vad["macro_f1"].iloc[0])
                if nf < vf:
                    parts.append(
                        f"NRC scores {nf:.3f} against VADER's {vf:.3f} — the richer lexicon "
                        f"is the weaker polarity method. NRC's value is its eight emotion "
                        f"categories, not its positive/negative split."
                    )
                else:
                    parts.append(f"NRC {nf:.3f} vs VADER {vf:.3f}.")
            d["sentiment"] = {
                "finding": " ".join(parts),
                "n_eval": f"{int(s['n_eval'].max()):,}",
            }

    # ---- irony ------------------------------------------------------------
    d["irony"] = {k: NA for k in
                  ("best_lexicon_acc", "baseline_acc", "best_acc", "best_name", "n_eval")}
    if bench is not None:
        i = bench[(bench["status"] == "ok") & (bench["task"] == "irony") &
                  (bench["corpus"] == "tweeteval_irony")]
        if len(i):
            base = i[i["method_key"] == "majority_baseline"]
            lex = i[i["family"] == "lexicon"]
            best = i[i["method_key"] != "majority_baseline"].sort_values("macro_f1", ascending=False)
            d["irony"] = {
                "best_lexicon_acc": f"{float(lex['accuracy'].max()):.3f}" if len(lex) else NA,
                "baseline_acc": f"{float(base['accuracy'].iloc[0]):.3f}" if len(base) else NA,
                "best_acc": f"{float(best['accuracy'].iloc[0]):.3f}" if len(best) else NA,
                "best_name": str(best["method_name"].iloc[0])[:38] if len(best) else NA,
                "n_eval": f"{int(i['n_eval'].max()):,}",
            }

    # ---- topics -----------------------------------------------------------
    if coh:
        bt, ld = coh["bertopic"]["npmi"], coh["lda"]["npmi"]
        verdict = "BERTopic wins" if bt > ld else "LDA wins" if ld > bt else "they tie"
        d["topics"] = {
            "summary": f"{coh['n_topics']} topics over {coh['n_documents']:,} captions, "
                       f"{coh['outlier_fraction']:.0%} outliers. On NPMI coherence, {verdict}: "
                       f"{bt:.3f} vs {ld:.3f}."
        }
    else:
        d["topics"] = {"summary": "Topic modelling did not run in this build."}

    # ---- model ------------------------------------------------------------
    d["model"] = {k: NA for k in ("r2", "ceiling_frac", "spearman", "ndcg", "baseline_index", "note")}
    d["ablation"] = {"negative": "Ablation did not run in this build."}
    d["price"] = {"negative": "Price model did not run in this build."}

    if models:
        p = models["performance"]
        d["model"] = {
            "r2": f"{p['r2_log']:.3f}",
            "ceiling_frac": f"{p['fraction_of_ceiling']:.0%}",
            "spearman": f"{p['spearman']:.3f}",
            "ndcg": f"{p['ndcg@10']:.3f}",
            "baseline_index": f"{p['baseline_composite_index']['r2_log']:.3f}",
            "note": (
                f"The Phase-1 weighted index is flattered here — isotonically calibrated on the "
                f"full dataset, an advantage the learned model does not get. It still loses by "
                f"{p['r2_log'] - p['baseline_composite_index']['r2_log']:.3f} R²."
            ),
        }
        drops = p.get("ablation", {}).get("drops", {})
        weak = [k for k, v in drops.items() if v["delta"] <= 0.005]
        if weak:
            worst = min(drops.items(), key=lambda kv: kv[1]["delta"])
            d["ablation"]["negative"] = (
                f"Removing the {', '.join(weak)} pillar does not hurt the model "
                f"({worst[1]['delta']:+.3f} R²). On this data the NLP features add no predictive "
                f"signal beyond reach, engagement and network. They still earn their place for "
                f"brand-safety screening and explanation — but not as predictors, and the report "
                f"does not claim otherwise."
            )
        elif drops:
            best = max(drops.items(), key=lambda kv: kv[1]["delta"])
            d["ablation"]["negative"] = (
                f"Every pillar contributes. The largest single contributor is {best[0]} "
                f"({best[1]['delta']:+.3f} R² when removed)."
            )

        q = models["price"]
        gap = q["r2_log"] - q["baseline_rate_card"]["r2_log"]
        d["price"]["negative"] = (
            f"The learned price model reaches R² {q['r2_log']:.3f} against "
            f"{q['baseline_rate_card']['r2_log']:.3f} for a published rate card with no learning "
            f"at all — a gap of {gap:.3f}. That does not justify the maintenance cost. Ship the "
            f"rule; revisit once real negotiated-deal data exists."
        )

    (DECK_DIR / "deck_data.json").write_text(json.dumps(d, indent=2))
    print(f"  deck data -> {DECK_DIR / 'deck_data.json'}")
    print(f"  figures   -> {len(list(figdir.glob('*.png')))} copied")
    return DECK_DIR / "deck_data.json"


if __name__ == "__main__":
    build()
