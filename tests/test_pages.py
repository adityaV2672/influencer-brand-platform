"""
Render every page, for both roles, and fail on any exception.

Screenshotting pages one at a time finds the page you happened to look at.
This finds all of them, and it is the check that runs before a deploy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT))

BRAND_PAGES = [
    "brand_overview", "brand_campaigns", "brand_discover", "brand_shortlist",
    "brand_requests", "brand_deals", "brand_reporting", "brand_builder",
]
CREATOR_PAGES = [
    "creator_overview", "creator_discover", "creator_requests", "creator_deals",
    "creator_analytics", "creator_earnings", "creator_profile",
]
# The four methods_* pages are no longer registered as routes, so they are no
# longer part of the product surface. Their files stay in views/ and are
# still smoke-tested below, because switching them back on is a one-line
# change in Home.py and a page that has rotted in the meantime is worse than
# no page at all.
SHARED_PAGES = ["settings", "help"]
RETIRED_PAGES = ["methods_model", "methods_nlp", "methods_network", "methods_data"]

CASES = ([("brand", p) for p in BRAND_PAGES + SHARED_PAGES]
         + [("creator", p) for p in CREATOR_PAGES + SHARED_PAGES])


def _run(role: str, page: str) -> AppTest:
    at = AppTest.from_file(str(APP / "views" / f"{page}.py"), default_timeout=90)
    at.session_state["role"] = role
    at.run()
    return at


@pytest.mark.parametrize("role,page", CASES, ids=[f"{r}:{p}" for r, p in CASES])
def test_page_renders(role, page):
    at = _run(role, page)
    assert not at.exception, (
        f"{role}/{page} raised: "
        + "\n".join(str(e.message) for e in at.exception)
    )


@pytest.mark.parametrize("page", RETIRED_PAGES)
def test_retired_page_still_renders(page):
    """Off the navigation, but not rotting."""
    at = _run("brand", page)
    assert not at.exception


def test_methods_are_not_routed():
    """The pages must be unreachable, not merely unlinked."""
    src = (APP / "Home.py").read_text(encoding="utf-8")
    active = [ln for ln in src.splitlines()
              if "methods_" in ln and not ln.strip().startswith("#")]
    assert not active, f"methods pages are still registered: {active}"


def test_signin_renders():
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=90)
    at.run()
    assert not at.exception
