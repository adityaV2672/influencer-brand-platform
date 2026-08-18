"""
Dashboard smoke test.

Runs every page headlessly through Streamlit's own test harness and fails on any
uncaught exception. This catches the class of bug that only appears when a page
actually renders - a missing column, a None where a DataFrame was expected, a
chart built from an empty frame - which no amount of import-checking finds.

    python -m tests.test_dashboard
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

PAGES = [
    ("Home (Discover)", "app/Home.py"),
    ("Creator profile", "app/pages/1_Creator_profile.py"),
    ("Brand matching", "app/pages/2_Brand_matching.py"),
    ("Model & methods", "app/pages/3_Model_and_methods.py"),
    ("Network map", "app/pages/4_Network_map.py"),
    ("Creator analytics", "app/pages/5_Creator_analytics.py"),
]


def run_page(label: str, rel: str, tier: str) -> tuple[bool, str]:
    from streamlit.testing.v1 import AppTest

    path = ROOT / rel
    if not path.exists():
        return False, f"missing file {rel}"
    try:
        at = AppTest.from_file(str(path), default_timeout=180)
        at.session_state["tier"] = tier
        at.run()
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"

    if at.exception:
        msgs = []
        for e in at.exception:
            msgs.append(f"{getattr(e, 'type', '?')}: {str(getattr(e, 'message', e))[:220]}")
        return False, " | ".join(msgs)

    n_widgets = len(at.get("dataframe")) + len(at.get("metric")) + len(at.get("markdown"))
    return True, f"rendered, {n_widgets} elements, {len(at.error)} error blocks"


def main() -> int:
    print("=" * 74)
    print("DASHBOARD SMOKE TEST")
    print("=" * 74)
    failures = []
    for tier in ("Free", "Paid"):
        print(f"\n--- {tier} plan ---")
        for label, rel in PAGES:
            ok, detail = run_page(label, rel, tier)
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:<22} {detail}")
            if not ok:
                failures.append(f"[{tier}] {label}: {detail}")

    print("-" * 74)
    if failures:
        print(f"{len(failures)} FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"all {len(PAGES) * 2} page renders passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
