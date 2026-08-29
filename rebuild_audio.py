"""
Rebuild the audio feature end to end, in the order the stages depend on.

    python rebuild_audio.py

Each stage is a separate process on purpose: the corpus is expensive to
regenerate and cheap to reuse, and a change to the prosody encoder invalidates
the saved embeddings, which is a mistake that is easy to make and hard to see
(the training stage will happily reuse stale embeddings and report unchanged
numbers).
"""
from __future__ import annotations

import subprocess
import sys

STAGES = [
    ("simulated corpus, annotators and ASR", ["-m", "src.audio.train", "--stage", "corpus"]),
    ("text / prosody / fusion", ["-m", "src.audio.train", "--stage", "train"]),
    ("sweep: ASR word error rate", ["-m", "src.audio.train", "--stage", "sweep-wer"]),
    ("sweep: recording noise", ["-m", "src.audio.train", "--stage", "sweep-noise"]),
    ("sweep: learning curve", ["-m", "src.audio.train", "--stage", "sweep-curve"]),
    ("product layer", ["-m", "src.features.export_nectar"]),
]


def main() -> int:
    for i, (label, args) in enumerate(STAGES, 1):
        print(f"\n[{i}/{len(STAGES)}] {label}")
        r = subprocess.run([sys.executable, *args])
        if r.returncode != 0:
            print(f"  stage failed: {label}")
            return r.returncode
    print("\nDone. Run: python -m pytest tests/ -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
