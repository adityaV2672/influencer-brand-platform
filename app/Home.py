"""
Nectar — influencer-brand collaboration platform.

Entry point. Sets up routing, injects the design system, draws the shell, and
hands off to the page. Streamlit's own multipage navigation is switched off
(`position="hidden"`) and replaced with the dark sidebar in nectar/shell.py,
because the product this replicates has a two-sided nav with a role switch and
Streamlit's default page list cannot express that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(
    page_title="Nectar",
    page_icon="🍯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from nectar import landing, shell, state  # noqa: E402

shell.inject_css()

V = "views"


def _page(script: str, title: str, url: str = "", icon: str = "") -> st.Page:
    """url="" marks the default page.

    Streamlit serves the FIRST page in the navigation list at the root URL and
    ignores any url_path given to it. Setting one anyway meant /brand-overview
    matched nothing, so every deep link opened Streamlit's "Page not found"
    dialog before falling back to the root page. Brand Overview is therefore
    the root, which is also correct: the root is the sign-in screen until a
    role is chosen, and the brand dashboard immediately after.
    """
    return st.Page(f"{V}/{script}.py", title=title,
                   url_path=url or None, icon=icon or None, default=not url)


# Icons are drawn from a single geometric family so the rail reads as one set.
BRAND_PAGES = [
    (_page("brand_overview", "Overview"), "Overview", ":material/dashboard:"),
    (_page("brand_campaigns", "Campaigns", "brand-campaigns"), "Campaigns", ":material/work:"),
    (_page("brand_builder", "Find creators", "brand-builder"), "Find creators", ":material/auto_awesome:"),
    (_page("brand_discover", "Discover", "brand-discover"), "Discover", ":material/search:"),
    (_page("brand_shortlist", "Shortlist", "brand-shortlist"), "Shortlist", ":material/bookmark:"),
    (_page("brand_requests", "Requests", "brand-requests"), "Requests", ":material/inbox:"),
    (_page("brand_deals", "Deals", "brand-deals"), "Deals", ":material/handshake:"),
    (_page("brand_reporting", "Reporting", "brand-reporting"), "Reporting", ":material/trending_up:"),
    (_page("onboarding_brand", "Set up", "brand-setup"), "Set up", ":material/tune:"),
]
BRAND_HIDDEN: list = []

CREATOR_PAGES = [
    (_page("creator_overview", "Overview", "creator-overview"), "Overview", ":material/dashboard:"),
    (_page("creator_discover", "Discover", "creator-discover"), "Discover", ":material/search:"),
    (_page("creator_requests", "Requests", "creator-requests"), "Requests", ":material/inbox:"),
    (_page("creator_deals", "Deals", "creator-deals"), "Deals", ":material/handshake:"),
    (_page("creator_analytics", "Analytics", "creator-analytics"), "Analytics", ":material/trending_up:"),
    (_page("creator_earnings", "Earnings", "creator-earnings"), "Earnings", ":material/payments:"),
    (_page("creator_profile", "Profile", "creator-profile"), "Profile", ":material/person:"),
    (_page("onboarding_creator", "Set up", "creator-setup"), "Set up", ":material/link:"),
]

# The Model & methods section is switched off in the product. The four page
# files remain in views/ and the block below restores them in one line if the
# validation evidence has to be shown again; they are simply not registered as
# routes, so there is no URL that reaches them either.
METHOD_PAGES: list = []
# METHOD_PAGES = [
#     (_page("methods_model", "Model", "methods-model"), "Model", ":material/insights:"),
#     (_page("methods_nlp", "NLP methods", "methods-nlp"), "NLP methods", ":material/psychology:"),
#     (_page("methods_network", "Network", "methods-network"), "Network", ":material/hub:"),
#     (_page("methods_data", "Data", "methods-data"), "Data", ":material/database:"),
# ]

FOOTER_PAGES = [
    (_page("metric_library", "Metric library", "metrics"), "Metric library", ":material/menu_book:"),
    (_page("settings", "Settings", "settings"), "Settings", ":material/settings:"),
    (_page("help", "Help", "help"), "Help", ":material/help:"),
]

ALL = ([p for p, _, _ in BRAND_PAGES] + BRAND_HIDDEN
       + [p for p, _, _ in CREATOR_PAGES]
       + [p for p, _, _ in METHOD_PAGES]
       + [p for p, _, _ in FOOTER_PAGES])

state.init()

# A role can be set from the URL (?role=brand). This is what makes a deep link
# to /brand-discover work when it is opened in a fresh session - without it the
# link lands on the sign-in screen and loses the route.
_qp_role = st.query_params.get("role")
if _qp_role in ("brand", "creator") and state.role() != _qp_role:
    state.set_role(_qp_role)

# Routes are registered BEFORE the role is checked. Registering them only for
# signed-in sessions meant a deep link opened in a fresh browser hit
# st.navigation-less code, and Streamlit answered with its "Page not found"
# dialog before the sign-in screen had a chance to appear.
nav = st.navigation(ALL, position="hidden")

# The sign-in screen is not a route: it is what the app shows before a role is
# chosen. Routing exists either way, so a deep link still lands correctly once
# a role is picked.
if state.role() is None:
    # The landing page IS the sign-in screen. A visitor who has never used
    # Nectar should meet the product's argument before its navigation, and the
    # role choice is the last thing on the page rather than the first.
    picked = landing.render()
    if picked:
        state.set_role(picked)
        st.rerun()
    st.stop()

st.session_state["_home_page"] = (
    "views/brand_overview.py" if state.role() == "brand" else "views/creator_overview.py"
)

shell.render_sidebar({
    "main": BRAND_PAGES if state.role() == "brand" else CREATOR_PAGES,
    "methods": METHOD_PAGES,
    "footer": FOOTER_PAGES,
})
nav.run()
