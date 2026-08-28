"""
Rebuild everything the audit's fixes touch, in dependency order.

Deliberately does NOT re-run generation pass 1, the NLP pipeline, the graph or
the brand-fit embedding pass. None of those depend on campaign outcomes, and
re-running the NLP alone would cost hours of transformer inference for an
identical result. What changes here is the campaign target, the models trained
on it, and everything downstream.

    python rebuild_after_audit.py
"""
from __future__ import annotations

import sys
import time

t0 = time.time()


def step(n, label):
    print(f"\n[{n}] {label}\n" + "-" * 74)


step(1, "Regenerating campaign outcomes (amplification now partly latent)")
from src.data.generate_synthetic import generate_campaigns_step
camps = generate_campaigns_step()
print(f"    {len(camps):,} campaigns")

step(2, "Rebuilding the modelling table")
from src.features.build_features import build_modelling_table, FEATURE_DIR
mt = build_modelling_table()
mt.to_parquet(FEATURE_DIR / "modelling_table.parquet", index=False)
print(f"    {mt.shape[0]:,} rows x {mt.shape[1]} columns")

step(3, "Retraining (nested early stopping, smearing, structural baseline)")
from src.models.train import run as train_run
res = train_run()

step(4, "Re-running the irony benchmark with tuned lexicon thresholds")
import pandas as pd
from src.benchmark.run_benchmarks import run_task, alignment_from_confusion, RESULTS_DIR
from dataclasses import asdict

fresh = []
for corpus in ("tweeteval_irony", "sarcasm_headlines"):
    fresh += run_task("irony", corpus, methods=["majority", "bing", "vader", "nrc"],
                      fast_sample=3000, verbose=True)
new = pd.DataFrame([asdict(r) for r in fresh])

old = pd.read_parquet(RESULTS_DIR / "results.parquet")
key = ["task", "corpus", "method_key"]
merged = pd.concat([old[~old.set_index(key).index.isin(new.set_index(key).index)], new],
                   ignore_index=True)

# Backfill the label-alignment diagnostic onto every row, including the ones
# recorded before the check existed. Permuting a confusion matrix's columns is
# exactly equivalent to relabelling the predictions, so these are not estimates.
merged["alignment"] = [
    alignment_from_confusion(c, list(l)) if isinstance(c, (list, tuple)) or hasattr(c, "__len__")
    else {}
    for c, l in zip(merged.confusion, merged.labels)
]
flagged = [f"{r.task}/{r.method_name}" for r in merged.itertuples()
           if (r.alignment or {}).get("label_alignment_suspect")]
merged.to_parquet(RESULTS_DIR / "results.parquet", index=False)
print(f"\n    {len(merged)} benchmark rows; label-alignment flags: "
      f"{flagged if flagged else 'none'}")

step(5, "Re-exporting the dashboard payload")
from src.features.export_app import run as export_app
export_app()

step(6, "Rebuilding the Nectar product layer")
from src.features.export_nectar import run as export_nectar
meta = export_nectar()

print(f"\nDone in {time.time() - t0:.0f}s")
