"""
End-to-end pipeline. One command rebuilds every artifact the dashboard reads.

    python run_pipeline.py                # everything
    python run_pipeline.py --skip nlp     # skip a stage
    python run_pipeline.py --only sna     # run one stage
    python run_pipeline.py --list         # show stages

Stage ordering is not arbitrary:

    generate  ->  posts and profiles exist
    sna       ->  needs posts (graph is built from co-hashtag / co-brand behaviour)
    campaigns ->  needs sna   (campaign outcomes depend on measured centrality)
    nlp       ->  needs posts (sentiment, emotion, topics, sarcasm per post)
    features  ->  needs all of the above merged into one influencer-level table
    models    ->  needs features + campaigns (the supervised target)
    export    ->  writes the slim artifacts the hosted dashboard loads

Every stage is idempotent and writes to artifacts/ or data/processed/.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections.abc import Callable

STAGES: list[tuple[str, str, Callable[[], object]]] = []


def stage(key: str, description: str):
    def deco(fn):
        STAGES.append((key, description, fn))
        return fn

    return deco


# ==========================================================================


@stage("generate", "Generate the synthetic influencer universe (profiles, posts, brands)")
def _generate():
    from src.data.generate_synthetic import generate

    return generate()


@stage("benchmarks", "Download real labelled NLP corpora (TweetEval, sarcasm headlines)")
def _benchmarks():
    from src.data.fetch_benchmarks import fetch_all

    return fetch_all()


@stage("sna", "Build the influencer graph and compute centrality features")
def _sna():
    from src.network.sna import run

    return run()


@stage("campaigns", "Simulate sponsored campaigns (the supervised target)")
def _campaigns():
    from src.data.generate_synthetic import generate_campaigns_step

    return generate_campaigns_step()


@stage("nlp", "Run the NLP pipeline over all posts (sentiment, emotion, topics, sarcasm)")
def _nlp():
    from src.nlp.pipeline import run

    return run()


@stage("features", "Merge every track into the influencer-level feature table")
def _features():
    from src.features.build_features import run

    return run()


@stage("models", "Train the performance and price models")
def _models():
    from src.models.train import run

    return run()


@stage("brandfit", "Build the brand-fit matrix (semantic similarity + safety gates)")
def _brandfit():
    from src.models.brandfit import run

    return run()


@stage("evaluate", "Evaluate NLP methods on the real labelled corpora")
def _evaluate():
    from src.benchmark.run_benchmarks import run_all

    return run_all()


@stage("export", "Export slim artifacts for the hosted dashboard")
def _export():
    from src.features.export_app import run

    return run()


@stage("figures", "Render the publication figures")
def _figures():
    from src.report.figures import build_all

    return build_all()


@stage("report", "Build the Word report and the slide-deck data")
def _report():
    from src.report.build_report import build
    from src.report.deck_data import build as build_deck_data

    path = build()
    build_deck_data()
    print(f"    report -> {path}")
    return path


@stage("verify", "Run the integrity checks")
def _verify():
    import subprocess
    import sys as _sys

    r = subprocess.run([_sys.executable, "-m", "tests.test_integrity"],
                       capture_output=True, text=True)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        raise RuntimeError("integrity checks FAILED - see output above")
    return True


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="run only these stages")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these stages")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'stage':<12} description")
        print("-" * 78)
        for key, desc, _ in STAGES:
            print(f"{key:<12} {desc}")
        return 0

    selected = [s for s in STAGES if (not args.only or s[0] in args.only) and s[0] not in args.skip]
    if not selected:
        print("No stages selected.")
        return 1

    print("=" * 78)
    print("INFLUENCER-BRAND COLLABORATION PLATFORM - pipeline")
    print(f"stages: {', '.join(s[0] for s in selected)}")
    print("=" * 78)

    failures = []
    t_all = time.time()
    for i, (key, desc, fn) in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {key.upper()} - {desc}")
        print("-" * 78)
        t0 = time.time()
        try:
            fn()
            print(f"-- {key} completed in {time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            failures.append((key, exc))
            print(f"!! {key} FAILED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
            if not args.continue_on_error:
                print("\nAborting. Re-run with --continue-on-error to push past this.")
                return 1

    print("\n" + "=" * 78)
    print(f"pipeline finished in {time.time() - t_all:.1f}s "
          f"({len(selected) - len(failures)}/{len(selected)} stages ok)")
    if failures:
        for key, exc in failures:
            print(f"  FAILED {key}: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
